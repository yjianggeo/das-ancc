# das-ancc

Ambient noise cross-correlation tomography for DAS (Distributed Acoustic Sensing) SAC data — extracts Rayleigh and Love wave phase-velocity dispersion curves and produces 2D lateral velocity maps.
基于 DAS（分布式声学传感）SAC 数据的背景噪声互相关层析成像——提取 Rayleigh 波和 Love 波相速度频散曲线，输出 2D 横向速度分布图。

**Pipeline / 流水线:** read SAC → preprocess → cross-correlate (ZZ + TT) → stack → FTAN dispersion → 2D tomography
读取 SAC → 预处理 → 互相关（ZZ + TT）→ 叠加 → FTAN 频散 → 2D 层析成像

🌐 **[English](#english)** | **[中文](#中文)**

---

<a name="english"></a>

## English

### Features

- Full ambient noise cross-correlation pipeline, from raw **SAC** files to 2D phase-velocity maps.
- **Rayleigh + Love joint processing**: uses DAS fiber-azimuth diversity (urban bent cables) to separate ZZ (Rayleigh) and TT (Love) cross-correlation components.
- **FTAN** (Frequency-Time Analysis) phase-velocity dispersion images with Hilbert-envelope picking.
- **Stage-by-stage resumable pipeline**: five independent stages, each persisted to HDF5 — any stage can be re-run without recomputing the others.
- **26 unit + integration tests** using synthetic data; no real DAS files required to run tests.

### Install

```bash
git clone https://github.com/yjianggeo/das-ancc.git
cd das-ancc
pip install -e .[dev]
```

Requires Python ≥ 3.10. Dependencies: `obspy`, `numpy`, `scipy`, `h5py`, `matplotlib`, `pandas`, `pyyaml`.

### Data layout

Organize one SAC file per channel, plus a cable geometry CSV:

```
data/
├── raw/
│   ├── CH000.sac
│   ├── CH001.sac
│   └── ...
└── cable_geometry.csv
```

**`cable_geometry.csv` format:**

```csv
channel,x_m,y_m,azimuth_deg,gauge_length_m
0,0.0,0.0,45.0,10.0
1,7.07,7.07,45.0,10.0
2,14.14,14.14,90.0,10.0
```

| Column | Description |
|---|---|
| `channel` | Integer channel ID |
| `x_m` | Easting in metres |
| `y_m` | Northing in metres |
| `azimuth_deg` | Fiber orientation, degrees clockwise from North |
| `gauge_length_m` | Gauge length in metres |

### Quick start

**1. Copy and edit the config:**

```bash
cp config/default.yaml config/myproject.yaml
```

**2. Run the full pipeline:**

```bash
python scripts/run_pipeline.py --config config/myproject.yaml
```

**3. Run a single stage** (for resuming or re-processing):

```bash
python scripts/run_pipeline.py --config config/myproject.yaml --stage preprocess
python scripts/run_pipeline.py --config config/myproject.yaml --stage correlate
python scripts/run_pipeline.py --config config/myproject.yaml --stage stack
python scripts/run_pipeline.py --config config/myproject.yaml --stage dispersion
python scripts/run_pipeline.py --config config/myproject.yaml --stage imaging
```

### Configuration reference

| Key | Meaning | Example |
|---|---|---|
| **`data`** | | |
| &nbsp;&nbsp;`sac_dir` | Directory containing SAC files | `"data/raw"` |
| &nbsp;&nbsp;`output_dir` | Where HDF5 outputs are written | `"data/processed"` |
| &nbsp;&nbsp;`geometry_file` | Path to cable geometry CSV | `"data/cable_geometry.csv"` |
| &nbsp;&nbsp;`components` | Components to compute | `["ZZ", "TT"]` |
| **`preprocess`** | | |
| &nbsp;&nbsp;`bandpass` | Bandpass corners in Hz | `[0.1, 5.0]` |
| &nbsp;&nbsp;`normalization` | Temporal normalization: `rms` \| `1bit` | `"rms"` |
| &nbsp;&nbsp;`whiten` | Spectral whitening (bool) | `true` |
| &nbsp;&nbsp;`segment_length` | Segment length in seconds | `3600` |
| **`correlate`** | | |
| &nbsp;&nbsp;`max_lag` | Max NCF lag in seconds | `60` |
| &nbsp;&nbsp;`min_distance` | Min inter-channel distance in m | `10` |
| &nbsp;&nbsp;`max_distance` | Max inter-channel distance in m | `2000` |
| **`stack`** | | |
| &nbsp;&nbsp;`method` | `linear` \| `pws` (phase-weighted) | `"linear"` |
| &nbsp;&nbsp;`min_segments` | Minimum segments required to stack | `10` |
| **`dispersion`** | | |
| &nbsp;&nbsp;`period_range` | Target period range in seconds | `[0.2, 5.0]` |
| &nbsp;&nbsp;`velocity_range` | Trial phase-velocity range in m/s | `[100, 1500]` |
| **`imaging`** | | |
| &nbsp;&nbsp;`grid_spacing` | Grid node spacing in metres | `50` |

### Output files

All outputs are written to `output_dir` as HDF5:

| File | Contents |
|---|---|
| `ncf.h5` | Raw cross-correlations per segment and channel pair |
| `stacked.h5` | Stacked NCFs per channel pair and component |
| `dispersion.h5` | Picked dispersion curves: `(period, velocity)` per pair |
| `imaging.h5` | 2D velocity maps per period and component |

### Visualization

```python
from das_ancc.viz.record_section import plot_record_section
from das_ancc.viz.dispersion_plot import plot_dispersion_image
from das_ancc.viz.velocity_map import plot_velocity_map

# NCF record section
fig = plot_record_section(ncfs, distances, dt=0.01, max_lag=30.0,
                           component="ZZ", output_path="record_section.png")

# FTAN dispersion image with picks
fig = plot_dispersion_image(image, periods, velocities,
                             picks=picks, output_path="dispersion.png")

# 2D lateral velocity map
fig = plot_velocity_map(velocity_map, x_grid, y_grid,
                         period=1.0, component="ZZ",
                         output_path="velocity_1s.png")
```

### Modules

| Module | Responsibility |
|---|---|
| `io` | SAC batch reading, cable geometry loader, HDF5 storage |
| `preprocess` | Detrend, bandpass, RMS/1-bit normalization, spectral whitening |
| `correlate` | FFT cross-correlation; fiber-azimuth ZZ/TT component projection |
| `stack` | Linear stacking and phase-weighted stacking (PWS) |
| `dispersion` | FTAN phase-velocity energy image; auto dispersion picking |
| `imaging` | Gaussian-kernel straight-ray tomography |
| `viz` | Record-section, dispersion image, and velocity-map plots |

### Algorithm notes

**Love wave extraction from DAS**

DAS measures axial strain rate along the fiber. For fiber azimuth θ and ray azimuth α between two channels, the component projection weights are:

- **ZZ (Rayleigh):** `w_ZZ = cos(θᵢ − α) × cos(θⱼ − α)`
- **TT (Love):** `w_TT = sin(θᵢ − α) × sin(θⱼ − α)`

Urban DAS cables with bends provide geometric diversity in fiber azimuth, enabling ZZ and TT component separation.

**FTAN phase velocity**

For each period T and trial velocity v, a Gaussian filter centred at f₀ = 1/T is applied to the NCF in the frequency domain. The Hilbert analytic envelope of the filtered signal is evaluated at lag t = dist/v. The resulting energy image peaks at the true phase velocity.

### Running tests

```bash
pytest tests/ -v
```

All 26 tests use synthetic data — no real DAS files required.

### License

MIT

---

<a name="中文"></a>

## 中文

### 功能特性

- 完整的背景噪声互相关流水线，从原始 **SAC** 文件直到 2D 相速度分布图。
- **Rayleigh + Love 联合处理**：利用 DAS 光缆方位角多样性（城市弯折光缆）分离 ZZ（Rayleigh）和 TT（Love）互相关分量。
- **FTAN**（频率-时间分析）相速度频散能量图，结合 Hilbert 包络自动拾取。
- **分阶段断点续跑**：五个独立阶段，每步结果持久化为 HDF5，任意阶段可单独重跑，无需重算其他步骤。
- **26 个单元和集成测试**，全部使用合成数据，无需真实 DAS 文件即可运行。

### 安装

```bash
git clone https://github.com/yjianggeo/das-ancc.git
cd das-ancc
pip install -e .[dev]
```

需要 Python ≥ 3.10。依赖：`obspy`、`numpy`、`scipy`、`h5py`、`matplotlib`、`pandas`、`pyyaml`。

### 数据组织

每个通道一个 SAC 文件，加上光缆几何文件：

```
data/
├── raw/
│   ├── CH000.sac
│   ├── CH001.sac
│   └── ...
└── cable_geometry.csv
```

**`cable_geometry.csv` 格式：**

```csv
channel,x_m,y_m,azimuth_deg,gauge_length_m
0,0.0,0.0,45.0,10.0
1,7.07,7.07,45.0,10.0
2,14.14,14.14,90.0,10.0
```

| 列 | 说明 |
|---|---|
| `channel` | 道号（整数）|
| `x_m` | 东向坐标（米）|
| `y_m` | 北向坐标（米）|
| `azimuth_deg` | 光缆方位角，从北顺时针（度）|
| `gauge_length_m` | 标距长度（米）|

### 快速开始

**1. 复制并编辑配置文件：**

```bash
cp config/default.yaml config/myproject.yaml
```

**2. 运行完整流水线：**

```bash
python scripts/run_pipeline.py --config config/myproject.yaml
```

**3. 运行单个阶段**（断点续跑或局部重处理）：

```bash
python scripts/run_pipeline.py --config config/myproject.yaml --stage preprocess
python scripts/run_pipeline.py --config config/myproject.yaml --stage correlate
python scripts/run_pipeline.py --config config/myproject.yaml --stage stack
python scripts/run_pipeline.py --config config/myproject.yaml --stage dispersion
python scripts/run_pipeline.py --config config/myproject.yaml --stage imaging
```

### 配置项说明

| 配置项 | 含义 | 示例 |
|---|---|---|
| **`data`** | | |
| &nbsp;&nbsp;`sac_dir` | SAC 文件目录 | `"data/raw"` |
| &nbsp;&nbsp;`output_dir` | HDF5 输出目录 | `"data/processed"` |
| &nbsp;&nbsp;`geometry_file` | 光缆几何文件路径 | `"data/cable_geometry.csv"` |
| &nbsp;&nbsp;`components` | 需计算的分量 | `["ZZ", "TT"]` |
| **`preprocess`** | | |
| &nbsp;&nbsp;`bandpass` | 带通频带（Hz）| `[0.1, 5.0]` |
| &nbsp;&nbsp;`normalization` | 时域归一化：`rms` \| `1bit` | `"rms"` |
| &nbsp;&nbsp;`whiten` | 谱白化（布尔）| `true` |
| &nbsp;&nbsp;`segment_length` | 时段长度（秒）| `3600` |
| **`correlate`** | | |
| &nbsp;&nbsp;`max_lag` | 最大互相关滞后（秒）| `60` |
| &nbsp;&nbsp;`min_distance` | 最小道间距（米）| `10` |
| &nbsp;&nbsp;`max_distance` | 最大道间距（米）| `2000` |
| **`stack`** | | |
| &nbsp;&nbsp;`method` | `linear` \| `pws`（相位加权叠加）| `"linear"` |
| &nbsp;&nbsp;`min_segments` | 最少叠加段数 | `10` |
| **`dispersion`** | | |
| &nbsp;&nbsp;`period_range` | 目标周期范围（秒）| `[0.2, 5.0]` |
| &nbsp;&nbsp;`velocity_range` | 相速度搜索范围（m/s）| `[100, 1500]` |
| **`imaging`** | | |
| &nbsp;&nbsp;`grid_spacing` | 成像网格间距（米）| `50` |

### 输出文件

所有输出写入 `output_dir`，格式为 HDF5：

| 文件 | 内容 |
|---|---|
| `ncf.h5` | 原始互相关（按时段和道对）|
| `stacked.h5` | 叠加互相关（按道对和分量）|
| `dispersion.h5` | 拾取的频散曲线：每道对 `(周期, 速度)` |
| `imaging.h5` | 各周期各分量的 2D 速度分布图 |

### 可视化

```python
from das_ancc.viz.record_section import plot_record_section
from das_ancc.viz.dispersion_plot import plot_dispersion_image
from das_ancc.viz.velocity_map import plot_velocity_map

# NCF 记录剖面
fig = plot_record_section(ncfs, distances, dt=0.01, max_lag=30.0,
                           component="ZZ", output_path="record_section.png")

# FTAN 频散能量图 + 拾取曲线
fig = plot_dispersion_image(image, periods, velocities,
                             picks=picks, output_path="dispersion.png")

# 2D 横向速度分布图
fig = plot_velocity_map(velocity_map, x_grid, y_grid,
                         period=1.0, component="ZZ",
                         output_path="velocity_1s.png")
```

### 模块

| 模块 | 职责 |
|---|---|
| `io` | SAC 批量读取、光缆几何加载、HDF5 存储 |
| `preprocess` | 去趋势、带通滤波、RMS/1-bit 归一化、谱白化 |
| `correlate` | FFT 互相关；光缆方位角 ZZ/TT 分量投影 |
| `stack` | 线性叠加与相位加权叠加（PWS）|
| `dispersion` | FTAN 相速度能量图；自动频散曲线拾取 |
| `imaging` | Gaussian 核直线射线走时层析 |
| `viz` | NCF 记录剖面、频散能量图、2D 速度分布图 |

### 算法说明

**DAS Love 波提取**

DAS 测量沿光缆方向的轴向应变率。设光缆方位角为 θ，道对间射线方位角为 α，分量投影权重为：

- **ZZ（Rayleigh）：** `w_ZZ = cos(θᵢ − α) × cos(θⱼ − α)`
- **TT（Love）：** `w_TT = sin(θᵢ − α) × sin(θⱼ − α)`

城市测道光缆的弯折提供了方位角多样性，使 ZZ 和 TT 分量可分离。

**FTAN 相速度**

对每个周期 T 和试验相速度 v，在频率域对互相关函数施加以 f₀ = 1/T 为中心的 Gaussian 滤波器，计算滤波后信号的 Hilbert 解析包络在 t = dist/v 处的幅值。能量图的极大值位置即为真实相速度。

### 运行测试

```bash
pytest tests/ -v
```

全部 26 个测试使用合成数据，无需真实 DAS 文件。

### 许可证

MIT
