# 退化数据集构建说明

本文档详细说明 MAR_SynCode 项目中退化数据集的构建流程。项目核心任务是 **金属伪影去除（Metal Artifact Reduction, MAR）**，通过在无伪影的 CT 图像上合成金属伪影来构建配对训练数据。

---

## 1. 整体架构

```
原始 CT 图像 (DeepLesion PNG)
        │
        ▼
┌─────────────────────────────┐
│  prepare_deep_lesion.py     │  ◄── 入口脚本
│  ├── 加载配置与物理参数       │
│  ├── 加载金属掩模             │
│  └── 遍历图像调用模拟函数     │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  simulate_data.py           │  ◄── 核心退化生成引擎
│  ├── 组织分解（水/骨）        │
│  ├── 正弦图投影               │
│  ├── 多能量转换 (keV→kVp)    │
│  ├── 泊松噪声模拟             │
│  ├── 射束硬化校正 (BHC)       │
│  ├── 金属伪影注入              │
│  └── FBP 重建 & 存储          │
└─────────────────────────────┘
              │
              ▼
   HDF5 配对数据集（gt.h5 + {idx}.h5）
              │
              ▼
┌─────────────────────────────┐
│  adn/datasets/deep_lesion.py│  ◄── PyTorch Dataset 加载
│  ├── 读取 HDF5               │
│  ├── 归一化到 [-1, 1]         │
│  └── 数据增强（翻转等）        │
└─────────────────────────────┘
```

---

## 2. 物理参数配置

### 2.1 CT 几何参数（`config/dataset_py_640geo.yaml`）

| 参数 | 值 | 含义 |
|------|-----|------|
| `imPixNum` | 416 | 重建图像像素尺寸 |
| `angNum` | 640 | 投影角度数 |
| `sinogram_size_x` | 640 | 正弦图 X 维度 |
| `sinogram_size_y` | 641 | 正弦图 Y 维度（探测器元素数） |
| `SOD` | 1075 mm | 射线源到物体距离 |
| `window` | [-175, 275] HU | CT 窗宽窗位 |

### 2.2 物理仿真参数（`util_func.py::get_mar_params()`）

| 参数 | 值 | 含义 |
|------|-----|------|
| `kev` | 70 | 单能量投影的参考能量 |
| `kVp` | 120 | 管电压 |
| `energies` | 20–120 keV | 能量网格范围 |
| `photonNum` | 2×10⁷ | 入射光子数 |
| `MiuWater` | 0.192 cm⁻¹ | 水的线衰减系数 |
| `threshWaterHU` | 100 HU | 水/混合组织阈值 |
| `threshBoneHU` | 1500 HU | 骨/混合组织阈值 |
| `scatterPhoton` | 20 | 散射光子数 |

### 2.3 金属材料

支持 4 种金属材料的质量衰减系数（默认使用钛 Titanium，`materialID=0`）：

| 材料 | 密度 (g/cm³) |
|------|-------------|
| Titanium (Ti) | 4.5 |
| Iron (Fe) | 7.8 |
| Copper (Cu) | 8.9 |
| Gold (Au) | 2.0 |

X 射线能谱来自 `GE14Spectrum120KVP.mat`。

---

## 3. 退化合成流程（`simulate_data.py`）

### 3.1 Step 1：组织成分分解

将 CT 图像分解为**水成分**和**骨成分**：

```python
# HU → 线衰减系数
img = imgCT / 1000 * MiuWater + MiuWater

# 阈值分割
bwWater = img <= threshWater      # 纯水区域
bwBone  = img >= threshBone       # 纯骨区域
bwBoth  = ~bwWater & ~bwBone      # 混合区域（线性插值分配）

# 混合区域中按比例分配
imgBone[bwBoth] = (img[bwBoth] - threshWater) / (threshBone - threshWater) * img[bwBoth]
imgWater[bwBoth] = img[bwBoth] - imgBone[bwBoth]
```

### 3.2 Step 2：正弦图投影

使用 ODL 库的扇束几何进行 Radon 变换（前向投影）：

```python
Pwater_kev = ray_trafo(imgWater)   # 水成分投影
Pbone_kev  = ray_trafo(imgBone)    # 骨成分投影
```

### 3.3 Step 3：多能量转换（keV → kVp）

调用 `pkev2kvp()` 将单能投影转换为多能投影，模拟真实 X 射线的多色性：

```python
# 对每个能量 ien，按能谱加权
for ien in energies:       # 20~120 keV
    for imat in range(matNum):
        projAll[:,:,imat] = MiuAll[ien]/MiuAll[kev] * projkevAll[:,:,imat]
    proj = sum(projAll, axis=2)
    ProjEnergy += spectrum[ien] * exp(-proj)

projkvp = -log(ProjEnergy / sum(spectrum))
```

### 3.4 Step 4：泊松噪声模拟

模拟光子计数过程中的量子噪声：

```python
temp = round(exp(-projkvp) * photonNum)  # 期望光子数
temp = temp + scatterPhoton               # 加散射光子（20个）
ProjPhoton = Poisson(temp)                # 泊松采样
projkvpNoise = -log(ProjPhoton / photonNum)
```

### 3.5 Step 5：射束硬化校正（BHC）

使用三阶多项式拟合进行水基 BHC：

```python
# 预计算的 BHC 多项式系数 paraBHC
p1BHC = [p1, p1², p1³] · paraBHC
poly_CT = FBP(p1BHC)    # 无金属、有噪声、已 BHC 的基准 CT
```

### 3.6 Step 6：金属伪影注入

对每个金属掩模执行以下操作：

#### (a) 金属投影与衰减
```python
Pmetal_kev = ray_trafo(imgMetal)           # 金属前向投影
Pmetal_kev = metalAtten * Pmetal_kev       # 乘以金属衰减系数
```

#### (b) 部分体积效应校正
```python
# 二值腐蚀找边缘像素
Pmetal_kev_bw = binary_erosion(Pmetal_kev > 0, structure=ones((1,3)))
Pmetal_edge = xor(Pmetal_kev > 0, Pmetal_kev_bw)
Pmetal_kev[Pmetal_edge] /= 4              # 边缘强度降至 25%
```

#### (c) 含金属正弦图合成
```python
projkevAll[:,:,2] = Pmetal_kev            # 水 + 骨 + 金属
projkvpMetal = pkev2kvp(...)              # 多能量转换
# + 泊松噪声（同 Step 4）
```

#### (d) 校正方法生成

生成多种校正结果作为训练数据：

| 输出 | 含义 |
|------|------|
| `ma_CT` | 含金属伪影的 CT 图像（FBP 重建） |
| `ma_sinogram` | 含金属伪影的正弦图 |
| `LI_CT` | 线性插值校正后的 CT 图像 |
| `LI_sinogram` | 线性插值校正后的正弦图 |
| `BHC_CT` | 射束硬化校正后的 CT 图像 |
| `BHC_sinogram` | 射束硬化校正后的正弦图 |
| `metal_trace` | 金属在正弦图中的投影轨迹（二值掩模） |

---

## 4. 数据存储格式

### 4.1 Ground Truth（`gt.h5`）

每个 CT 切片生成一个 `gt.h5`，包含：

| 字段 | 类型 | 含义 |
|------|------|------|
| `image` | float32 | 无伪影 GT 图像（线衰减系数域） |
| `poly_sinogram` | float32 | BHC 后的基准正弦图 |
| `poly_CT` | float32 | BHC + 泊松噪声后的基准 CT |

### 4.2 金属伪影数据（`{idx}.h5`）

每个金属掩模生成一个 HDF5 文件：

| 字段 | 类型 | 含义 |
|------|------|------|
| `ma_CT` | float32 | 含伪影 CT |
| `LI_CT` | float32 | 线性插值校正 CT |
| `BHC_CT` | float32 | BHC 校正 CT |
| `ma_sinogram` | float32 | 含伪影正弦图 |
| `LI_sinogram` | float32 | 插值校正正弦图 |
| `BHC_sinogram` | float32 | BHC 校正正弦图 |
| `metal_trace` | uint8 | 金属投影轨迹掩模 |

### 4.3 目录结构

```
dataset_dir/
├── train_640geo/
│   ├── <patient_id>/<slice_id>/
│   │   ├── gt.h5           # Ground truth
│   │   ├── 0.h5            # 金属掩模 0 的伪影数据
│   │   ├── 1.h5            # 金属掩模 1 的伪影数据
│   │   └── ...
│   └── ...
└── test_640geo/
    └── ...（同上）
```

---

## 5. 训练/测试集划分

在 `config/dataset_py_640geo.yaml` 中定义：

| 划分 | 图像数 | 索引规则 | 金属掩模 |
|------|--------|----------|---------|
| 训练集 | 1000 张 | `np.arange(0,1000)*40` | 90 个（排除测试用掩模） |
| 测试集 | 200 张 | `np.arange(0,200)*10 + 44999` | 10 个（索引 0,1,19,29,35,42,62,63,97,99） |

---

## 6. 数据加载与预处理（`adn/datasets/deep_lesion.py`）

### 6.1 归一化流程

```python
# 值域裁剪到 [0.0, 0.5]，然后映射到 [-1, 1]
data = clip(data, 0.0, 0.5)
data = (data - 0.0) / (0.5 - 0.0)      # → [0, 1]
data = data * 2.0 - 1.0                 # → [-1, 1]
```

### 6.2 反归一化

```python
data = data * 0.5 + 0.5                 # → [0, 1]
data = data * (0.5 - 0.0) + 0.0         # → [0.0, 0.5]
```

### 6.3 HU → 线衰减系数转换

```python
image = image - hu_offset               # 减去 HU 偏移 (32768)
image[image < -1000] = -1000             # 裁剪下界
image = image / 1000 * 0.192 + 0.192     # HU → 线衰减系数
```

### 6.4 数据增强

| 增强方式 | 训练 | 测试 |
|---------|------|------|
| 随机水平翻转 | ✓ | ✗ |
| 随机选择金属掩模 | ✓ | ✗ |
| 部分保留（非配对训练） | 可选 (50%) | ✗ |

### 6.5 输出格式

```python
{
    "data_name": str,      # 文件标识
    "hq_image": Tensor,    # 高质量（无伪影）图像, shape: [1, H, W]
    "lq_image": Tensor,    # 低质量（含伪影）图像, shape: [1, H, W]
    "mask": Tensor          # 金属掩模（可选）, shape: [1, H, W]
}
```

---

## 7. 其他数据集

项目还支持两种额外的数据集格式：

### 7.1 Spineweb（`adn/datasets/spineweb.py`）

- 格式：NumPy `.npy` 文件
- 输入：配对的伪影/无伪影脊柱 CT 图像
- 归一化范围：`[-1000, 2000]` HU → `[-1, 1]`

### 7.2 NatureImage（`adn/datasets/nature_image.py`）

- 格式：标准图像文件
- 输入：配对的伪影/无伪影自然图像
- 预处理：Resize(384) → RandomCrop(256) → [0,255] → [-1,1]

---

## 8. 关键依赖

| 库 | 用途 |
|-----|------|
| [ODL](https://github.com/odlgroup/odl) | CT 正/反向投影（扇束几何、FBP 重建） |
| `scipy` | 二值腐蚀、线性插值、矩阵运算 |
| `h5py` | HDF5 数据读写 |
| `numpy` | 数值计算 |
| `PyTorch` | Dataset/DataLoader |

---

## 9. 退化类型总结

本项目模拟的退化因素包括：

1. **金属伪影**：高密度金属植入物导致的条状伪影
2. **射束硬化效应**：多色 X 射线穿过物体后低能光子优先被吸收
3. **泊松量子噪声**：X 射线光子计数的统计涨落
4. **部分体积效应**：金属边缘像素的不完全投影
5. **散射光子**：探测器接收到的散射辐射（固定 20 光子）
