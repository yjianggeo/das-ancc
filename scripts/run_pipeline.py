"""DAS Ambient Noise Cross-Correlation Tomography -- CLI entry point."""
import argparse
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from das_ancc.config import load_config
from das_ancc.correlate.component_projection import (
    inter_station_azimuth, projection_weights,
)
from das_ancc.correlate.cross_correlate import fft_correlate
from das_ancc.dispersion.ftan import ftan_phase_velocity
from das_ancc.dispersion.picking import auto_pick
from das_ancc.imaging.tomography import gaussian_tomography
from das_ancc.io.das_metadata import load_geometry
from das_ancc.io.sac_reader import read_sac_segments
from das_ancc.io.writer import (
    load_ncfs, save_dispersion, save_ncf,
    save_stacked_ncf, save_velocity_map,
    load_stacked_ncf, load_dispersion,
)
from das_ancc.preprocess.pipeline import preprocess_stream
from das_ancc.stack.stacking import stack_ncfs

STAGES = ["preprocess", "correlate", "stack", "dispersion", "imaging"]


def stage_preprocess(cfg, segments, geom):
    pp = []
    for seg_id, st in segments:
        st_pp = preprocess_stream(
            st,
            bandpass=tuple(cfg.preprocess["bandpass"]),
            normalization=cfg.preprocess["normalization"],
            whiten=cfg.preprocess["whiten"],
        )
        pp.append((seg_id, st_pp))
    return pp


def stage_correlate(cfg, preprocessed, geom, output_dir):
    fs = preprocessed[0][1][0].stats.sampling_rate
    max_lag_s = cfg.correlate["max_lag"]
    max_lag_samples = int(max_lag_s * fs)
    channels = geom.channel.tolist()

    for seg_id, st in preprocessed:
        # Map station name suffix to channel index
        data = {}
        for tr in st:
            try:
                ch = int(tr.stats.station.removeprefix("CH"))
            except ValueError:
                continue
            data[ch] = tr.data

        for i, j in combinations(channels, 2):
            if i not in data or j not in data:
                continue
            row_i = geom[geom.channel == i].iloc[0]
            row_j = geom[geom.channel == j].iloc[0]
            dist = np.hypot(row_j.x_m - row_i.x_m, row_j.y_m - row_i.y_m)
            if dist < cfg.correlate["min_distance"] or dist > cfg.correlate["max_distance"]:
                continue
            ray_az = inter_station_azimuth(row_i.x_m, row_i.y_m, row_j.x_m, row_j.y_m)
            w_zz, w_tt = projection_weights(row_i.azimuth_deg, row_j.azimuth_deg, ray_az)
            raw_ncf = fft_correlate(data[i], data[j], max_lag_samples)
            if "ZZ" in cfg.data["components"] and abs(w_zz) > 0.1:
                save_ncf(output_dir, seg_id, i, j, "ZZ", raw_ncf * w_zz)
            if "TT" in cfg.data["components"] and abs(w_tt) > 0.1:
                save_ncf(output_dir, seg_id, i, j, "TT", raw_ncf * w_tt)


def stage_stack(cfg, geom, output_dir):
    channels = geom.channel.tolist()
    for comp in cfg.data["components"]:
        for i, j in combinations(channels, 2):
            ncfs = load_ncfs(output_dir, i, j, comp)
            stacked = stack_ncfs(ncfs, method=cfg.stack["method"],
                                  min_segments=cfg.stack["min_segments"])
            if stacked is not None:
                save_stacked_ncf(output_dir, i, j, comp, stacked)


def stage_dispersion(cfg, geom, output_dir):
    channels = geom.channel.tolist()
    pmin, pmax = cfg.dispersion["period_range"]
    vmin, vmax = cfg.dispersion["velocity_range"]
    periods = np.linspace(pmin, pmax, 30)
    velocities = np.linspace(vmin, vmax, 60)
    max_lag_s = cfg.correlate["max_lag"]

    for comp in cfg.data["components"]:
        for i, j in combinations(channels, 2):
            row_i = geom[geom.channel == i].iloc[0]
            row_j = geom[geom.channel == j].iloc[0]
            dist = np.hypot(row_j.x_m - row_i.x_m, row_j.y_m - row_i.y_m)
            ncf = load_stacked_ncf(output_dir, i, j, comp)
            if ncf is None:
                continue
            dt = max_lag_s / (len(ncf) // 2)
            image = ftan_phase_velocity(ncf, dt=dt, dist=dist,
                                         periods=periods, velocities=velocities)
            picks = auto_pick(image, periods, velocities)
            save_dispersion(output_dir, i, j, comp, picks)


def stage_imaging(cfg, geom, output_dir):
    channels = geom.channel.tolist()
    pmin, pmax = cfg.dispersion["period_range"]
    periods = np.linspace(pmin, pmax, 30)

    x_coords = geom.x_m.values
    y_coords = geom.y_m.values
    spacing = cfg.imaging["grid_spacing"]
    x_grid = np.arange(x_coords.min(), x_coords.max() + spacing, spacing)
    y_grid = np.arange(y_coords.min(), y_coords.max() + spacing, spacing)
    if len(y_grid) == 0:
        y_grid = np.array([y_coords.mean()])
    sigma = spacing * 2

    for comp in cfg.data["components"]:
        for ip, period in enumerate(periods):
            measurements = []
            for i, j in combinations(channels, 2):
                picks = load_dispersion(output_dir, i, j, comp)
                if picks is None or ip >= len(picks):
                    continue
                v = picks[ip, 1]
                if v <= 0:
                    continue
                row_i = geom[geom.channel == i].iloc[0]
                row_j = geom[geom.channel == j].iloc[0]
                measurements.append({
                    "x1": row_i.x_m, "y1": row_i.y_m,
                    "x2": row_j.x_m, "y2": row_j.y_m,
                    "velocity": v,
                })
            if measurements:
                vmap = gaussian_tomography(measurements, x_grid, y_grid, sigma)
                save_velocity_map(output_dir, period, comp, vmap)


def main():
    parser = argparse.ArgumentParser(
        description="DAS Ambient Noise Cross-Correlation Tomography"
    )
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument(
        "--stage",
        choices=STAGES + ["all"],
        default="all",
        help="Pipeline stage to run (default: all)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_dir = cfg.data["output_dir"]
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    geom = load_geometry(cfg.data["geometry_file"])
    stages_to_run = STAGES if args.stage == "all" else [args.stage]

    segments = None
    preprocessed = None

    if "preprocess" in stages_to_run or "correlate" in stages_to_run:
        print("Reading SAC files...")
        segments = read_sac_segments(cfg.data["sac_dir"],
                                      cfg.preprocess["segment_length"])
        print(f"  {len(segments)} segments found")

    if "preprocess" in stages_to_run:
        print("Stage: preprocess")
        preprocessed = stage_preprocess(cfg, segments, geom)

    if "correlate" in stages_to_run:
        print("Stage: correlate")
        if preprocessed is None:
            preprocessed = stage_preprocess(cfg, segments, geom)
        stage_correlate(cfg, preprocessed, geom, output_dir)

    if "stack" in stages_to_run:
        print("Stage: stack")
        stage_stack(cfg, geom, output_dir)

    if "dispersion" in stages_to_run:
        print("Stage: dispersion")
        stage_dispersion(cfg, geom, output_dir)

    if "imaging" in stages_to_run:
        print("Stage: imaging")
        stage_imaging(cfg, geom, output_dir)

    print("Done.")


if __name__ == "__main__":
    main()
