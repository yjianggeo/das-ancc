# das-ancc

**DAS Ambient Noise Cross-Correlation Tomography**
**DAS 背景噪声互相关层析成像**

A modular Python pipeline for ambient noise cross-correlation tomography using Distributed Acoustic Sensing (DAS) SAC data. Extracts Rayleigh and Love wave phase-velocity dispersion curves and produces 2D lateral velocity maps.

基于分布式声学传感（DAS）SAC 格式数据的模块化背景噪声互相关层析成像 Python 流水线，可提取 Rayleigh 波和 Love 波相速度频散曲线，并生成 2D 横向速度分布图。

---

## Features / 功能特性

- **Rayleigh + Love joint processing** / **Rayleigh + Love 联合处理**：利用光缆方位角多样性（城市弯折光缆）同时提取 ZZ 和 TT 分量互相关
- **Stage-by-stage resumable pipeline** / **分阶段断点续跑**：五个独立阶段均持久化到 HDF5，任意阶段可单独重跑
- **Modular design** / **模块化设计**：每个模块单一职责，算法可独立替换
- **26 unit + integration tests** / **26 个单元和集成测试**：合成数据全程覆盖

---

## Pipeline Overview / 流水线总览

```
SAC files
   ↓ [io]          Read + attach cable geometry (x, y, azimuth, gauge_length)
   ↓ [preprocess]  Detrend → bandpass → RMS normalize → spectral whiten
   ↓ [correlate]   FFT cross-correlation → ZZ / TT component projection
   ↓ [stack]       Linear or phase-weighted stacking (PWS)
   ↓ [dispersion]  FTAN phase-velocity image → auto-pick dispersion curves
   ↓ [imaging]     Gaussian-kernel straight-ray tomography
   2D lateral velocity maps (per period)
```

---

## Installation / 安装

Requires Python 3.10+.  需要 Python 3.10 及以上。

```bash
git clone https://github.com/yjianggeo/das-ancc.git
cd das-ancc
pip install -e ".[dev]"
```

Dependencies / 依赖项：`obspy`, `numpy`, `scipy`, `h5py`, `matplotlib`, `pandas`, `pyyaml`

---

## Quick Start / 快速开始

### 1. Prepare data / 准备数据

Organize your SAC files and create a cable geometry CSV:

准备 SAC 文件，并创建光缆几何文件：

```
data/
├── raw/                   # SAC files, one file per channel
│   ├── CH000.sac
│   ├── CH001.sac
│   └── ...
└── cable_geometry.csv     # Channel coordinates and fiber orientation
```

**`cable_geometry.csv` format / 格式：**

```csv
channel,x_m,y_m,azimuth_deg,gauge_length_m
0,0.0,0.0,45.0,10.0
1,7.07,7.07,45.0,10.0
2,14.14,14.14,90.0,10.0
...
```

| Column / 列 | Description / 说明 |
|---|---|
| `channel` | Channel ID (integer) / 道号（整数）|
| `x_m` | Easting in metres / 东向坐标（米）|
| `y_m` | Northing in metres / 北向坐标（米）|
| `azimuth_deg` | Fiber orientation, degrees clockwise from North / 光缆方位角，从北顺时针（度）|
| `gauge_length_m` | Gauge length in metres / 标距长度（米）|

### 2. Configure / 配置

Copy and edit the default config:  复制并修改默认配置：

```bash
cp config/default.yaml config/myproject.yaml
```

Key parameters / 关键参数：

```yaml
data:
  sac_dir: "data/raw"             # SAC file directory / SAC 文件目录
  output_dir: "data/processed"    # Output directory / 输出目录
  geometry_file: "data/cable_geometry.csv"
  components: ["ZZ", "TT"]       # ZZ=Rayleigh, TT=Love

preprocess:
  bandpass: [0.1, 5.0]           # Frequency range in Hz / 频率范围（Hz）
  normalization: "rms"            # "rms" or "1bit"
  whiten: true
  segment_length: 3600            # Segment length in seconds / 时段长度（秒）

correlate:
  max_lag: 60                     # Max NCF lag in seconds / 最大滞后时间（秒）
  min_distance: 10                # Min inter-station distance in m / 最小道间距（米）
  max_distance: 2000              # Max inter-station distance in m / 最大道间距（米）

stack:
  method: "linear"                # "linear" or "pws"
  min_segments: 10                # Min segments required to stack / 最少叠加段数

dispersion:
  period_range: [0.2, 5.0]       # Period range in seconds / 目标周期范围（秒）
  velocity_range: [100, 1500]     # Trial velocity range in m/s / 相速度搜索范围（m/s）

imaging:
  grid_spacing: 50                # Grid node spacing in m / 成像网格间距（米）
```

### 3. Run / 运行

Run the full pipeline:  运行完整流水线：

```bash
python scripts/run_pipeline.py --config config/myproject.yaml
```

Run a single stage (for resuming or re-processing):  运行单个阶段（用于断点续跑或重新处理）：

```bash
python scripts/run_pipeline.py --config config/myproject.yaml --stage preprocess
python scripts/run_pipeline.py --config config/myproject.yaml --stage correlate
python scripts/run_pipeline.py --config config/myproject.yaml --stage stack
python scripts/run_pipeline.py --config config/myproject.yaml --stage dispersion
python scripts/run_pipeline.py --config config/myproject.yaml --stage imaging
```

---

## Output Files / 输出文件

All outputs are written to `output_dir` as HDF5 files:

所有输出写入 `output_dir`，格式为 HDF5：

| File / 文件 | Contents / 内容 |
|---|---|
| `ncf.h5` | Raw cross-correlations per segment and channel pair / 原始互相关（按时段和道对）|
| `stacked.h5` | Stacked NCFs per channel pair and component / 叠加互相关（按道对和分量）|
| `dispersion.h5` | Picked dispersion curves: `(period, velocity)` per pair / 拾取的频散曲线 |
| `imaging.h5` | 2D velocity maps per period and component / 各周期 2D 速度分布图 |

---

## Visualization / 可视化

```python
import numpy as np
import h5py
from das_ancc.viz.record_section import plot_record_section
from das_ancc.viz.dispersion_plot import plot_dispersion_image
from das_ancc.viz.velocity_map import plot_velocity_map

# NCF record section / NCF 记录剖面
# (load your stacked NCFs and distances, then:)
fig = plot_record_section(ncfs, distances, dt=0.01, max_lag=30.0,
                           component="ZZ", output_path="record_section.png")

# Dispersion image with picks / 频散能量图 + 拾取曲线
fig = plot_dispersion_image(image, periods, velocities,
                             picks=picks, output_path="dispersion.png")

# 2D velocity map / 2D 速度分布图
fig = plot_velocity_map(velocity_map, x_grid, y_grid,
                         period=1.0, component="ZZ",
                         output_path="velocity_map_1s.png")
```

---

## Project Structure / 项目结构

```
das-ancc/
├── config/
│   └── default.yaml          # Default configuration / 默认配置
├── das_ancc/                  # Main package / 主包
│   ├── config.py              # Config loader / 配置加载
│   ├── io/
│   │   ├── sac_reader.py      # SAC batch reader + segmentation
│   │   ├── das_metadata.py    # Cable geometry loader
│   │   └── writer.py          # HDF5 storage
│   ├── preprocess/
│   │   ├── normalize.py       # RMS / 1-bit normalization
│   │   ├── whiten.py          # Spectral whitening
│   │   └── pipeline.py        # Preprocessing pipeline
│   ├── correlate/
│   │   ├── cross_correlate.py        # FFT cross-correlation
│   │   └── component_projection.py   # ZZ/TT azimuth projection
│   ├── stack/
│   │   └── stacking.py        # Linear + PWS stacking
│   ├── dispersion/
│   │   ├── ftan.py            # FTAN phase-velocity image
│   │   └── picking.py         # Auto dispersion picking
│   ├── imaging/
│   │   └── tomography.py      # Gaussian-kernel tomography
│   └── viz/
│       ├── record_section.py  # NCF record section plot
│       ├── dispersion_plot.py # Dispersion image plot
│       └── velocity_map.py    # 2D velocity map plot
├── scripts/
│   └── run_pipeline.py        # CLI entry point
└── tests/                     # 26 tests (synthetic data)
```

---

## Algorithm Notes / 算法说明

### Love Wave Extraction from DAS / DAS Love 波提取

DAS 测量沿光缆方向的轴向应变率。设光缆方位角为 θ，道对间射线方位角为 α，则：

DAS measures axial strain rate along the fiber. For fiber azimuth θ and ray azimuth α between two channels:

- **ZZ weight (Rayleigh):** `w_ZZ = cos(θᵢ − α) × cos(θⱼ − α)`
- **TT weight (Love):** `w_TT = sin(θᵢ − α) × sin(θⱼ − α)`

Urban DAS cables with bends provide geometric diversity in fiber azimuth, enabling separation of ZZ and TT components.

城市测道光缆弯折提供方位角多样性，使 ZZ 和 TT 分量可分离。

### FTAN Phase Velocity / FTAN 相速度

For each period T and trial velocity v, a Gaussian filter centred at f₀ = 1/T is applied to the NCF in the frequency domain. The Hilbert analytic envelope of the filtered signal is evaluated at lag t = dist/v. The resulting energy image is peaked at the true phase velocity.

对每个周期 T 和试验相速度 v，在频率域对互相关函数施加以 f₀ = 1/T 为中心的 Gaussian 滤波器，提取 Hilbert 解析信号包络在 t = dist/v 处的幅值。能量图的极大值位置即为真实相速度。

---

## Running Tests / 运行测试

```bash
pytest tests/ -v
```

All tests use synthetic data — no real DAS files required.

所有测试使用合成数据，无需真实 DAS 文件。

---

## License / 许可证

MIT
