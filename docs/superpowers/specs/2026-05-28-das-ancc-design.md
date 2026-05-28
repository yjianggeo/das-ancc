# DAS 背景噪声互相关层析成像程序设计文档

**日期**：2026-05-28  
**项目**：das-ancc (DAS Ambient Noise Cross-Correlation)  
**语言**：Python 3.10+  
**场景**：城市/浅层 DAS（测道/管线光缆），100–1000 道，数天至数周连续数据

---

## 1. 目标与范围

构建一个模块化流水线程序，对 DAS SAC 格式数据执行背景噪声互相关层析成像，最终输出：

- 每道对的 Rayleigh 波（ZZ 分量）和 Love 波（TT 分量）频散曲线
- 沿测线的 2D 横向相速度分布图（各周期）

不包含在本版本范围内：

- 3D 体层析反演
- 实时流处理
- 深度学习辅助拾取

---

## 2. 技术依赖

| 用途 | 库 |
|---|---|
| SAC 文件读写 | ObsPy |
| 数组运算 / FFT | NumPy, SciPy |
| 并行处理 | concurrent.futures (ProcessPoolExecutor) |
| 配置文件 | PyYAML |
| 可视化 | Matplotlib |
| 存储格式 | NumPy `.npy` / HDF5 (h5py) |
| 测试 | pytest |

---

## 3. 项目目录结构

```
das-ancc/
├── config/
│   └── default.yaml            # 默认配置
├── das_ancc/                   # 主 Python 包
│   ├── __init__.py
│   ├── io/
│   │   ├── __init__.py
│   │   ├── sac_reader.py       # 批量读取 SAC，按时段切分
│   │   ├── das_metadata.py     # 光缆几何文件读取，附加道元数据
│   │   └── writer.py           # NCF / 频散曲线结果存盘
│   ├── preprocess/
│   │   ├── __init__.py
│   │   ├── normalize.py        # 时间域归一化（1-bit / RMS）
│   │   ├── whiten.py           # 谱白化
│   │   └── pipeline.py         # 预处理步骤组合器
│   ├── correlate/
│   │   ├── __init__.py
│   │   ├── cross_correlate.py  # FFT 互相关，输出原始 NCF
│   │   └── component_projection.py  # DAS 应变率 → ZZ/TT 分量投影
│   ├── stack/
│   │   ├── __init__.py
│   │   └── stacking.py         # 线性叠加 / 相位加权叠加（PWS）
│   ├── dispersion/
│   │   ├── __init__.py
│   │   ├── ftan.py             # FTAN 频散能量图计算
│   │   └── picking.py          # 自动频散曲线拾取
│   ├── imaging/
│   │   ├── __init__.py
│   │   └── tomography.py       # 直线射线走时层析（Gaussian 核正则化）
│   └── viz/
│       ├── __init__.py
│       ├── record_section.py   # NCF 记录剖面图
│       ├── dispersion_plot.py  # 频散能量图 + 拾取曲线叠加
│       └── velocity_map.py     # 2D 速度横向分布图
├── scripts/
│   └── run_pipeline.py         # CLI 入口（argparse）
├── tests/
│   ├── conftest.py             # 合成数据 fixtures
│   ├── test_preprocess.py
│   ├── test_correlate.py
│   ├── test_dispersion.py
│   └── test_imaging.py
├── docs/
│   └── superpowers/specs/
│       └── 2026-05-28-das-ancc-design.md
└── pyproject.toml
```

---

## 4. 数据流

```
SAC 文件（原始 DAS 数据）
   ↓ io.sac_reader       读取 + 按时段切分（默认 1 小时/段）
   ↓ io.das_metadata     附加 (x, y, azimuth, gauge_length) 到每道
ObsPy Stream（含 DAS 元数据）
   ↓ preprocess.pipeline 去趋势 → 带通滤波 → 时间归一化 → 谱白化
预处理后的 Stream
   ↓ correlate.component_projection  按道方位角投影到 ZZ / TT 分量
   ↓ correlate.cross_correlate       FFT 互相关，按时段输出
原始 NCF（每时段 × 每道对 × 分量）
   ↓ stack.stacking      线性或相位加权叠加，过滤低质量时段
叠加 NCF（每道对 × 分量）
   ↓ dispersion.ftan     窄带 Gaussian 滤波组，生成频散能量图
   ↓ dispersion.picking  自动拾取相速度极大值，输出频散曲线
(道对, 周期, 相速度) 测量表
   ↓ imaging.tomography  直线射线走时反演，Gaussian 核平滑
2D 横向速度分布图（各周期，单位 m/s）
```

---

## 5. DAS Love 波提取方案

DAS 测量沿光缆方向的轴向应变率。设光缆方位角为 θ，Rayleigh 波粒子运动垂直于地面（对 ZZ 分量），Love 波粒子运动水平且垂直于传播方向（对 TT 分量）。

对于城市测道光缆，光缆在平面内存在弯折，不同道的方位角 θ 各异。利用这一几何多样性，`component_projection.py` 将各道的应变率按方位角分解，重建 ZZ（Rayleigh）和 TT（Love）互相关分量。

**前提**：需要准确的光缆逐道方位角文件（`cable_geometry.csv`），格式为：

```csv
channel, x_m, y_m, azimuth_deg, gauge_length_m
0, 0.0, 0.0, 45.0, 10.0
1, 7.07, 7.07, 45.0, 10.0
...
```

---

## 6. 配置文件结构

```yaml
# config/default.yaml
data:
  sac_dir: "data/raw"
  output_dir: "data/processed"
  geometry_file: "data/cable_geometry.csv"
  components: ["ZZ", "TT"]

preprocess:
  bandpass: [0.1, 5.0]        # Hz
  normalization: "rms"        # "1bit" | "rms"
  whiten: true
  segment_length: 3600        # 秒

correlate:
  max_lag: 60                 # 秒
  min_distance: 10            # 米
  max_distance: 2000          # 米

stack:
  method: "linear"            # "linear" | "pws"
  min_segments: 10

dispersion:
  period_range: [0.2, 5.0]   # 秒
  velocity_range: [100, 1500] # m/s

imaging:
  grid_spacing: 50            # 米
  regularization: 0.1

parallel:
  n_workers: 4
```

---

## 7. CLI 接口

```bash
# 运行完整流水线
python scripts/run_pipeline.py --config config/default.yaml

# 单独运行某一步（支持断点续跑）
python scripts/run_pipeline.py --config config/default.yaml --stage preprocess
python scripts/run_pipeline.py --config config/default.yaml --stage correlate
python scripts/run_pipeline.py --config config/default.yaml --stage stack
python scripts/run_pipeline.py --config config/default.yaml --stage dispersion
python scripts/run_pipeline.py --config config/default.yaml --stage imaging
```

每步结果写入 `output_dir/<stage>/`，下一步直接从磁盘读取，支持任意断点续跑。

---

## 8. 并行策略

- `correlate` 和 `preprocess` 步骤按时段（每小时）并行，使用 `ProcessPoolExecutor(n_workers)`
- `dispersion` 按道对并行
- `imaging` 单进程（数据量小，矩阵运算为主）

---

## 9. 测试策略

所有测试使用合成数据，不依赖真实 DAS 文件。

| 测试 | 方法 |
|---|---|
| `test_preprocess` | 生成白噪声 Stream，验证谱白化后功率谱平坦度 |
| `test_correlate` | 生成已知延迟的双道平面波，验证互相关峰位置误差 < 1 采样 |
| `test_dispersion` | 合成单频散曲线信号，验证 FTAN 拾取误差 < 5% |
| `test_imaging` | 双异常体合成走时，验证反演速度误差 < 5% |

---

## 10. 里程碑（供实现计划参考）

1. 项目脚手架（目录结构、`pyproject.toml`、配置加载）
2. `io` 模块：SAC 读取 + 几何元数据
3. `preprocess` 模块 + 单元测试
4. `correlate` 模块（ZZ 分量）+ 单元测试
5. `correlate` 模块（TT 分量投影）
6. `stack` 模块 + 集成测试（预处理 → 互相关 → 叠加）
7. `dispersion` 模块（FTAN + 自动拾取）+ 单元测试
8. `imaging` 模块（走时层析）+ 单元测试
9. `viz` 模块（三类图）
10. CLI 集成 + 完整流水线集成测试
