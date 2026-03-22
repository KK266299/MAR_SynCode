# 给 Cursor 的提示词：从零构建 CT 金属伪影退化数据集

> 直接将下方 `---` 之间的全部内容复制粘贴给 Cursor。

---

我需要你帮我构建一个用于 CT 金属伪影去除（Metal Artifact Reduction, MAR）的退化数据集。
目标是：给定干净的 CT 图像和金属掩模，通过物理仿真合成含金属伪影的退化图像，形成配对的 (GT, 退化图像) 数据集。

不完整的部分请参考原始仓库 https://github.com/liaohaofu/adn ，特别是其中的 MATLAB 实现。

请严格按照以下方法步骤构建。

================================================================
一、整体流水线概述
================================================================

输入：
  - 干净 CT 图像（HU 值域，如 DeepLesion 的 16-bit PNG）
  - 金属掩模（二值图，标注金属植入物区域，多个掩模存于一个 .mat 文件）
  - 物理参数（材料衰减系数 .mat、X 射线能谱 .mat）

输出（每张 CT × 每个金属掩模 → 一组 HDF5 文件）：
  - gt.h5：包含 ground truth CT（线衰减系数域）、基准正弦图、基准重建
  - {mask_idx}.h5：包含 ma_CT（含伪影）、LI_CT（线性插值校正）、BHC_CT（射束硬化校正）、对应正弦图、金属轨迹掩模

流水线 7 个步骤：
  1. HU → 线衰减系数转换
  2. 组织分解（水 + 骨）
  3. 前向投影（Radon 变换）→ 正弦图
  4. 单能 → 多能转换（keV → kVp）
  5. 泊松噪声仿真
  6. 射束硬化校正（BHC）
  7. 金属伪影注入 + 多种校正方法生成

================================================================
二、CT 几何与前向/反向投影
================================================================

使用 ODL 库构建扇束（Fan-Beam）CT 几何，参数如下：

  图像尺寸：416 × 416 像素
  像素分辨率：reso = 512/416 * 0.03 ≈ 0.0369 cm/pixel
  物理尺寸：sx = sy = 416 * reso
  投影角度数：640（0 到 2π 均匀分布）
  探测器元素数：641
  源到物体距离（SOD）：1075 * reso
  探测器到中心距离（DDE）：1075 * reso
  探测器总长度：su = 2 * sqrt(sx² + sy²)

构建方法：
```python
import odl
reco_space = odl.uniform_discr(
    min_pt=[-sx/2, -sy/2], max_pt=[sx/2, sy/2],
    shape=[416, 416], dtype='float32')
angle_partition = odl.uniform_partition(0, 2*np.pi, 640)
detector_partition = odl.uniform_partition(-su/2, su/2, 641)
geometry = odl.tomo.FanBeamGeometry(
    angle_partition, detector_partition,
    src_radius=1075*reso, det_radius=1075*reso)
ray_trafo = odl.tomo.RayTransform(reco_space, geometry, impl='astra_cuda')
FBPOper = odl.tomo.fbp_op(ray_trafo, filter_type='Ram-Lak', frequency_scaling=1.0)
```

ray_trafo(image) 做前向投影（图像 → 正弦图）。
FBPOper(sinogram) 做滤波反投影重建（正弦图 → 图像）。

================================================================
三、物理参数加载
================================================================

需要从 .mat 文件加载以下物理参数：

材料衰减系数（质量衰减系数表，覆盖 1~120 keV，多种衰减模式列）：
  - MiuofH2O.mat       → 水
  - MiuofBONE_Cortical_ICRU44.mat → 皮质骨
  - MiuofTi.mat        → 钛
  - MiuofFe.mat        → 铁
  - MiuofCu.mat        → 铜
  - MiuofAu.mat        → 金

X 射线能谱：
  - GE14Spectrum120KVP.mat → 120kVp 管电压下的 GE 球管能谱
    取第 2 列（索引 1）作为 spectrum

关键常数：
  kVp = 120                    # 管电压
  kev = 70                     # 参考单能量
  energies = np.arange(20, 121)  # 能量网格 20~120 keV
  photonNum = 2e7              # 入射光子数
  MiuWater = 0.192 cm⁻¹       # 70keV 下水的线衰减系数

组织分割阈值（先转为线衰减系数域）：
  threshWaterHU = 100 HU   → threshWater = 100/1000 * 0.192 + 0.192 = 0.2112
  threshBoneHU = 1500 HU   → threshBone = 1500/1000 * 0.192 + 0.192 = 0.48

金属材料参数（默认使用 Titanium, materialID=0）：
  densityMetal = [4.5, 7.8, 8.9, 2.0]   # Ti, Fe, Cu, Au 密度 (g/cm³)
  metalAtten = densityMetal[0] * MiuofMetal[kev-1, 6, 0]
  # 其中 MiuofMetal 由 4 种金属的衰减系数沿第 3 维 stack 而成
  # 列索引 6 对应 AttenuMode=7（total attenuation with coherent scattering）

水基 BHC 多项式系数预计算：
```python
thickness = np.arange(0, 50.01, 0.05).reshape(-1, 1)  # 水厚度 (cm)
pwaterkev = MiuofH2O[kev-1, 6] * thickness             # 单能投影
pwaterkvp = pkev2kvp(pwaterkev, spectrum, energies, kev, MiuofH2O[:kVp, :])  # 多能投影
A = np.concatenate([pwaterkvp, pwaterkvp**2, pwaterkvp**3], axis=1)
paraBHC = np.linalg.pinv(A) @ pwaterkev                # 三阶多项式拟合系数
```

================================================================
四、核心退化合成方法（逐步详解）
================================================================

### Step 1：HU → 线衰减系数

原始 DeepLesion PNG 是 16-bit 无符号整数，需要转换：
```python
image = raw_png * 65536 - 32768    # 恢复 HU 值（带偏移）
image = resize(image, (416, 416))  # 双线性插值缩放
image[image < -1000] = -1000       # 裁剪下界（空气以下无意义）
```

### Step 2：组织成分分解

将 CT 图像分解为水成分和骨成分，用于独立建模不同材料的衰减：
```python
img = imgCT / 1000 * MiuWater + MiuWater   # HU → 线衰减系数

imgWater = np.zeros_like(img)
imgBone = np.zeros_like(img)

bwWater = (img <= threshWater)                    # 纯水区域
bwBone = (img >= threshBone)                      # 纯骨区域
bwBoth = ~bwWater & ~bwBone                       # 混合区域

imgWater[bwWater] = img[bwWater]                  # 水区直接赋值
imgBone[bwBone] = img[bwBone]                     # 骨区直接赋值
# 混合区域按线性插值分配骨/水比例：
imgBone[bwBoth] = (img[bwBoth] - threshWater) / (threshBone - threshWater) * img[bwBoth]
imgWater[bwBoth] = img[bwBoth] - imgBone[bwBoth]
```

### Step 3：前向投影

对水和骨分别做 Radon 变换，得到各自的正弦图：
```python
Pwater_kev = ray_trafo(imgWater)   # shape: (640, 641)
Pbone_kev = ray_trafo(imgBone)     # shape: (640, 641)
```

### Step 4：单能 → 多能投影转换（pkev2kvp）

这是物理仿真的核心。真实 X 射线是多色的（包含多个能量），不同能量下材料的衰减系数不同。

方法：对每个能量 ien（20~120 keV），将参考能量 kev=70 下的投影按衰减系数比值缩放，再按能谱加权求和：

```python
def pkev2kvp(projkevAll, spectrum, energies, kev, MiuAll):
    """
    projkevAll: shape (views, bins, num_materials), 每种材料在 kev 下的投影
    spectrum: shape (120,), X射线能谱强度分布
    energies: array [20, 21, ..., 120], 能量网格
    kev: int, 参考能量（70）
    MiuAll: shape (120, num_modes, num_materials), 质量衰减系数表
    """
    AttenuMode = 7  # 使用第 7 列（total attenuation with coherent scattering）
    ProjEnergy = 0
    for ien in energies:
        for imat in range(num_materials):
            # 当前能量下的投影 = 参考能量投影 × (当前能量衰减 / 参考能量衰减)
            projAll[:,:,imat] = MiuAll[ien-1, AttenuMode-1, imat] / MiuAll[kev-1, AttenuMode-1, imat] * projkevAll[:,:,imat]
        proj_total = sum(projAll, axis=2)              # 合并所有材料
        ProjEnergy += spectrum[ien-1] * exp(-proj_total)  # Beer-Lambert 定律 + 能谱加权

    projkvp = -log(ProjEnergy / sum(spectrum[energies-1]))  # 归一化后取负对数
    return projkvp
```

### Step 5：泊松噪声仿真

模拟 X 射线探测器的光子计数统计噪声：
```python
scatterPhoton = 20                                    # 散射光子常数
expected_photons = round(exp(-projkvp) * photonNum)   # 期望到达光子数
expected_photons += scatterPhoton                     # 加散射
actual_photons = np.random.poisson(expected_photons)  # 泊松采样
actual_photons[actual_photons == 0] = 1               # 避免 log(0)
projkvpNoise = -log(actual_photons / photonNum)       # 含噪声的投影
```

### Step 6：水基射束硬化校正（BHC）

使用预计算的三阶多项式系数对投影做 BHC：
```python
p1 = projkvpNoise.reshape(-1, 1)
p1BHC = np.concatenate([p1, p1**2, p1**3], axis=1) @ paraBHC   # 三阶多项式
poly_sinogram = p1BHC.reshape(views, bins)
poly_CT = FBPOper(poly_sinogram)   # FBP 重建 → 无金属、有噪声、已BHC的基准CT
```

这个 poly_CT 连同原始线衰减系数图 gt_CT 一起存为 gt.h5。

### Step 7：金属伪影注入

对每个金属掩模执行以下操作：

#### 7a. 金属前向投影 + 衰减
```python
imgMetal = resize(metal_mask, (416, 416))   # 缩放到 CT 尺寸
Pmetal_kev = ray_trafo(imgMetal)            # 前向投影
metal_trace = (Pmetal_kev > 0)              # 金属在正弦图中的轨迹
Pmetal_kev *= metalAtten                    # 乘以金属线衰减系数
```

#### 7b. 部分体积效应（Partial Volume Effect）校正
金属边缘像素只占部分体素，衰减不应为 100%：
```python
Pmetal_bw = binary_erosion(Pmetal_kev > 0, structure=np.ones((1, 3)))  # 腐蚀
Pmetal_edge = np.logical_xor(Pmetal_kev > 0, Pmetal_bw)               # 边缘 = 原始 XOR 腐蚀
Pmetal_kev[Pmetal_edge] /= 4                                          # 边缘衰减降至 25%
```

#### 7c. 含金属正弦图合成
```python
projkevAll[:, :, 2] = Pmetal_kev          # 第 3 个材料通道 = 金属
projkvpMetal = pkev2kvp(projkevAll, ...)   # 水 + 骨 + 金属的多能投影
# 再加泊松噪声（与 Step 5 相同方法）
```

#### 7d. 三种校正结果生成

(1) 直接重建（含严重伪影）：
```python
p1 = projkvpMetalNoise.reshape(-1, 1)
ma_sinogram = (np.concatenate([p1, p1**2, p1**3], axis=1) @ paraBHC).reshape(views, bins)
ma_CT = FBPOper(ma_sinogram)
```

(2) 线性插值（LI）校正：
正弦图中金属轨迹处的值用相邻非金属位置的值线性插值替代：
```python
def interpolate_projection(proj, metalTrace):
    Pinterp = proj.copy()
    for each row i:
        metalpos = where(metalTrace[i] == 1)
        nonmetalpos = where(metalTrace[i] == 0)
        Pinterp[i, metalpos] = interp1d(nonmetalpos, proj[i, nonmetalpos])(metalpos)
    return Pinterp

LI_sinogram = interpolate_projection(ma_sinogram, metal_trace)
LI_CT = FBPOper(LI_sinogram)
```

(3) MAR 射束硬化校正（marBHC）：
```python
def marBHC(proj, metalBW, ray_trafo, FBPOper):
    projMetal = ray_trafo(metalBW)                  # 金属前向投影
    Pinterp = interpolate_projection(proj, projMetal > 0)  # LI 校正
    projDiff = proj - Pinterp                       # 残差 = 原始 - LI

    # 对金属投影区域做三阶多项式最小二乘拟合
    A[:, 0] = projMetal_masked
    A[:, 1] = projMetal_masked ** 2
    A[:, 2] = projMetal_masked ** 3
    X0 = lstsq(A, projDiff_masked)                  # 拟合系数

    # 用拟合结果校正
    projDelta = X0[0]*projMetal - polyval(X0, projMetal)
    projBHC = proj + projDelta
    imBHC = FBPOper(projBHC)
    return imBHC, projBHC
```

================================================================
五、输出数据格式
================================================================

每张 CT 图像生成一个子目录，结构如下：

```
output_dir/<patient>/<slice>/
├── gt.h5
│   ├── image          (float32) 无伪影 GT，线衰减系数域
│   ├── poly_sinogram  (float32) BHC 基准正弦图
│   └── poly_CT        (float32) 基准重建 CT
├── 0.h5               (第 0 个金属掩模)
│   ├── ma_CT          (float32) 含伪影 CT
│   ├── LI_CT          (float32) LI 校正 CT
│   ├── BHC_CT         (float32) BHC 校正 CT
│   ├── ma_sinogram    (float32) 含伪影正弦图
│   ├── LI_sinogram    (float32) LI 校正正弦图
│   ├── BHC_sinogram   (float32) BHC 校正正弦图
│   └── metal_trace    (uint8)   金属投影轨迹二值掩模
├── 1.h5               (第 1 个金属掩模)
└── ...
```

所有 HDF5 数据使用 gzip 压缩。

================================================================
六、数据集加载（PyTorch Dataset）
================================================================

训练时的加载与归一化方法：

值域：线衰减系数 [0.0, 0.5] → 归一化到 [-1, 1]
```python
# 归一化
data = np.clip(data, 0.0, 0.5)
data = (data - 0.0) / (0.5 - 0.0) * 2.0 - 1.0

# 反归一化
data = data * 0.5 + 0.5
data = data * (0.5 - 0.0) + 0.0
```

GT 读取 gt.h5 中的 'image' 字段。
退化图像读取 {idx}.h5 中的 'ma_CT' 字段。
两者已在线衰减系数域，无需额外转换。

数据增强：训练时随机水平翻转 + 随机选择金属掩模索引。

================================================================
七、依赖库
================================================================

pip install odl numpy scipy h5py pillow tqdm pyyaml

ODL 用于 CT 前向/反向投影（需要 ASTRA-Toolbox 的 CUDA 支持）。
如果没有 GPU，可以将 impl='astra_cuda' 改为 impl='astra_cpu' 或使用 ODL 内置实现。

================================================================
八、参考原始实现
================================================================

以上方法改写自 https://github.com/liaohaofu/adn 的 MATLAB 实现。
如有不清楚的地方，请参考该仓库中的：
  - +helper/simulate_metal_artifact.m  → 伪影合成主函数
  - +helper/pkev2kvp.m                 → 多能量转换
  - +helper/interpolate_projection.m   → 线性插值校正
  - +helper/get_mar_params.m           → 物理参数加载
  - prepare_deep_lesion.m              → 数据准备入口
  - config/dataset.yaml                → 配置参数

---
