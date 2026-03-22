# Cursor 提示词：构建 MAR 退化数据集

> 将以下内容完整粘贴给 Cursor，让它帮你实现数据集构建。

---

## 提示词正文

```
你现在需要帮我完成 MAR（Metal Artifact Reduction）项目的退化数据集构建。
本项目的目标是：在无金属伪影的干净 CT 图像上，通过物理仿真合成含金属伪影的退化图像，
形成配对数据集用于训练去伪影网络。

项目仓库已有部分代码（基于 https://github.com/liaohaofu/adn 的 Python 改写版），
但存在以下问题需要你补全和修正。请严格按照以下说明完成。

================================================================
一、项目结构总览
================================================================

MAR_SynCode/
├── simulate_data.py          # 核心：金属伪影合成引擎（已完成）
├── util_func.py              # 核心：物理仿真工具函数（已完成）
├── prepare_deep_lesion.py    # 训练集生成入口（已完成，需修正路径）
├── prepare_deep_lesion_texst.py  # 测试集生成入口（文件名拼写错误，需重命名）
├── gene_h5list.py            # H5 文件索引生成（有 bug 需修复）
├── config/
│   ├── dataset_py_640geo.yaml   # 数据集参数配置
│   ├── dataset.yaml             # 本地路径版配置
│   └── adn.yaml                 # 训练配置
├── adn/
│   ├── build_gemotry.py         # CT 扇束几何构建（依赖 ODL）
│   ├── build_gemotry_geo.py     # 可变投影数版本
│   ├── build_gemotry_imagesize.py  # 可变图像尺寸版本
│   ├── datasets/
│   │   ├── __init__.py          # Dataset 工厂
│   │   ├── deep_lesion.py       # DeepLesion PyTorch Dataset
│   │   ├── spineweb.py          # Spineweb PyTorch Dataset
│   │   └── nature_image.py      # NatureImage PyTorch Dataset
│   └── utils/
│       ├── __init__.py
│       └── misc.py              # 通用工具
├── data/
│   └── deep_lesion/
│       ├── image_list.txt       # 337,898 条图像路径
│       ├── blacklist.json       # 排除列表
│       └── metal_masks/         # 金属掩模 + 物理参数（.mat 文件）
│           ├── SampleMasks.mat
│           ├── MiuofH2O.mat
│           ├── MiuofTi.mat / MiuofFe.mat / MiuofCu.mat / MiuofAu.mat
│           ├── MiuofBONE_Cortical_ICRU44.mat
│           └── GE14Spectrum120KVP.mat
└── docs/

================================================================
二、需要你完成的具体任务（按优先级排序）
================================================================

【任务 1】修复 gene_h5list.py 的语法错误
--------------------------------------------------------------
文件：gene_h5list.py，第 7 行
问题：`os. listdir` 中间有一个空格，应改为 `os.listdir`
修复后确保脚本可以正常运行。

【任务 2】重命名 prepare_deep_lesion_texst.py → prepare_deep_lesion_test.py
--------------------------------------------------------------
- 将文件重命名为 prepare_deep_lesion_test.py
- 确认内部逻辑使用 `['test']` 作为 splits（当前已是）

【任务 3】修正 config/dataset_py_640geo.yaml 中的路径
--------------------------------------------------------------
当前 raw_dir 和 dataset_dir 指向腾讯云外部存储路径（不可用）：
  raw_dir: /apdcephfs/share_1290796/hazelhwang/mardataset/Images_png
  dataset_dir: /apdcephfs_cq3/share_1290796/hazelhwang/mardataset

修改为本地相对路径：
  raw_dir: data/deep_lesion/raw        # 用户需软链接到 DeepLesion Images_png
  dataset_dir: data/deep_lesion        # 输出目录

【任务 4】编写 prepare_spineweb.py（参考原始仓库）
--------------------------------------------------------------
原始仓库 https://github.com/liaohaofu/adn 中有 prepare_spineweb.py，
其功能是从 Spineweb 原始 MHD/NII.GZ 医学影像中提取切片，
按是否含金属伪影分为 artifact / no_artifact 两类，最终输出 .npy 文件。

请严格参考以下逻辑实现：

```python
# 核心流程：
# 1. 读取 config/dataset.yaml 中的 spineweb 配置
# 2. 遍历 raw_dir 下所有 patient 目录
# 3. 对每个 volume（.mhd 或 .nii.gz）用 SimpleITK 读取
# 4. 逐切片判断是否含金属伪影：
#    - image.max() > max_hu[1]（默认 2500）→ 检测连通区域
#      - 最大连通区域面积 > connected_area（默认 400 像素）→ "artifact"
#      - 否则跳过（小金属碎片，不可用）
#    - image.max() > max_hu[0]（默认 2000）但 <= max_hu[1] → 跳过（边界情况）
#    - image.max() <= max_hu[0] → "no_artifact"
# 5. Resize 到 image_size（默认 256×256）
# 6. 保存为 .npy 文件，同时生成缩略图 .png
# 7. 最后划分 train/test：
#    - 随机打乱 patient 目录
#    - 累计收集 num_tests（默认 200）张图像的 patient 作为测试集
#    - 其余为训练集
#    - 将文件移动到 train/artifact、train/no_artifact、test/artifact、test/no_artifact
```

需要的依赖：SimpleITK, torch (用于 make_grid 生成概览图), PIL, numpy, tqdm
使用 adn.utils 中已有的 read_dir() 和 get_connected_components() 函数。

【任务 5】确保 DeepLesion Dataset 类支持 H5 格式
--------------------------------------------------------------
当前 adn/datasets/deep_lesion.py 中 load_data() 方法（第 92-95 行）使用的是 .mat 格式：
```python
def load_data(self, data_file):
    gt = sio.loadmat(data_file[0])['image']
    metal = sio.loadmat(data_file[1])['image']
    return self.convert2coefficient(gt).T, metal
```

但 simulate_data.py 生成的数据是 .h5 (HDF5) 格式。需要修改 DeepLesion 类以支持 H5：

1. 修改 __init__ 中的文件扫描逻辑：
   - 将 predicate 从匹配 "gt.mat" 改为同时支持 "gt.mat" 和 "gt.h5"
   - 将 metal_files 的匹配从 .mat 扩展为同时支持 .mat 和 .h5

2. 修改 load_data() 方法：
```python
def load_data(self, data_file):
    gt_file, metal_file = data_file
    if gt_file.endswith('.h5'):
        import h5py
        with h5py.File(gt_file, 'r') as f:
            gt = f['image'][()]       # 线衰减系数域，无需再转换
        with h5py.File(metal_file, 'r') as f:
            metal = f['ma_CT'][()]    # 含伪影的 CT
        return gt, metal
    else:
        gt = sio.loadmat(gt_file)['image']
        metal = sio.loadmat(metal_file)['image']
        return self.convert2coefficient(gt).T, metal
```

注意：H5 格式的数据在 simulate_data.py 中已经转换为线衰减系数域（img = imgCT/1000*MiuWater+MiuWater），
所以读取 H5 时不需要再调用 convert2coefficient()。

【任务 6】编写完整的数据集构建运行脚本 build_dataset.sh
--------------------------------------------------------------
编写一个 shell 脚本，整合所有步骤：

```bash
#!/bin/bash
set -e

echo "=== MAR Dataset Construction Pipeline ==="

# Step 0: 检查依赖
python -c "import odl; import torch; import h5py; import SimpleITK" || {
    echo "缺少依赖，请先安装："
    echo "  pip install odl torch h5py SimpleITK scipy numpy pillow tqdm pyyaml"
    exit 1
}

# Step 1: 检查原始数据链接
if [ ! -d "data/deep_lesion/raw" ]; then
    echo "请先创建 DeepLesion 数据软链接："
    echo "  ln -s /path/to/DeepLesion/Images_png data/deep_lesion/raw"
    exit 1
fi

# Step 2: 生成 DeepLesion 训练集
echo "[1/5] 生成 DeepLesion 训练集..."
python prepare_deep_lesion.py

# Step 3: 生成 DeepLesion 测试集
echo "[2/5] 生成 DeepLesion 测试集..."
python prepare_deep_lesion_test.py

# Step 4: 生成 H5 文件索引
echo "[3/5] 生成训练集 H5 索引..."
python gene_h5list.py --h5_image data/deep_lesion/train_640geo --h5_list data/deep_lesion/train_640geo_dir.txt

echo "[4/5] 生成测试集 H5 索引..."
python gene_h5list.py --h5_image data/deep_lesion/test_640geo --h5_list data/deep_lesion/test_640geo_dir.txt

# Step 5: 准备 Spineweb（如有原始数据）
if [ -d "data/spineweb/raw" ]; then
    echo "[5/5] 生成 Spineweb 数据集..."
    python prepare_spineweb.py
else
    echo "[5/5] 跳过 Spineweb（未找到 data/spineweb/raw）"
fi

echo "=== 数据集构建完成 ==="
```

================================================================
三、核心退化合成流程（供你理解，不需修改）
================================================================

simulate_data.py 中 simulate_metal_artifact() 的完整流程：

1. 组织分解：将 CT 图像按 HU 阈值分解为水成分(imgWater)和骨成分(imgBone)
   - 阈值：水 ≤ 100 HU (对应衰减系数 0.2112)，骨 ≥ 1500 HU (对应 0.48)
   - 混合区域按线性插值分配

2. 正弦图投影：用 ODL 的扇束几何对水和骨分别做前向投影 (Radon 变换)
   - 416×416 图像 → 640×641 正弦图

3. 多能量转换 (keV→kVp)：调用 pkev2kvp()
   - 70keV 单能投影 → 120kVp 多色投影
   - 按 GE14Spectrum120KVP 能谱加权

4. 泊松噪声：模拟 2×10^7 光子 + 20 散射光子的量子噪声
   - ProjPhoton = Poisson(round(exp(-proj) * photonNum) + 20)
   - projNoise = -log(ProjPhoton / photonNum)

5. 射束硬化校正 (BHC)：三阶多项式拟合
   - 使用预计算的水基 BHC 系数 paraBHC

6. 无金属基准：poly_CT = FBP(BHC后正弦图)，存为 gt.h5

7. 金属伪影注入（对每个金属掩模并行执行）：
   a) 金属前向投影 + 衰减系数缩放 (metalAtten)
   b) 部分体积效应：边缘像素衰减降至 25%
   c) 合成含金属正弦图 = 水 + 骨 + 金属投影 → pkev2kvp → 泊松噪声
   d) 生成 3 种校正版本：
      - ma_CT：直接 FBP 重建（含严重伪影）
      - LI_CT：线性插值校正后 FBP
      - BHC_CT：marBHC 射束硬化校正后
   e) 保存为 {idx}.h5

util_func.py 中的关键函数（不需修改）：
- pkev2kvp(): 单能→多能投影转换
- interpolate_projection(): 正弦图线性插值（金属区域用相邻非金属值填充）
- marBHC(): 一阶多项式射束硬化校正
- get_mar_params(): 加载所有物理参数（材料衰减、能谱、BHC 系数等）

================================================================
四、物理参数速查（不需修改，仅供参考）
================================================================

CT 几何（adn/build_gemotry.py）:
  图像尺寸: 416×416
  投影角度: 640
  探测器元素: 641
  源到物体距离: 1075 mm
  重建方法: FBP (Ram-Lak 滤波器)

物理常数（util_func.py::get_mar_params()）:
  参考能量 kev = 70
  管电压 kVp = 120
  能量范围 = 20~120 keV
  入射光子数 = 2×10^7
  散射光子数 = 20
  水衰减系数 MiuWater = 0.192 cm^-1
  默认金属 = Titanium (密度 4.5 g/cm³, materialID=0)

支持金属材料:
  | ID | 材料 | 密度 (g/cm³) |
  | 0  | Ti   | 4.5          |
  | 1  | Fe   | 7.8          |
  | 2  | Cu   | 8.9          |
  | 3  | Au   | 2.0          |

训练/测试划分:
  训练: 1000 张 CT (indices = np.arange(0,1000)*40)，90 个金属掩模
  测试: 200 张 CT (indices = np.arange(0,200)*10+44999)，10 个金属掩模

================================================================
五、不完整的部分请参考原始仓库
================================================================

如果以上信息不够完整，请参考：https://github.com/liaohaofu/adn

重点参考文件：
- prepare_deep_lesion.m（MATLAB 版数据准备，本项目已改写为 Python）
- prepare_spineweb.py（Spineweb 数据准备，本项目需新建）
- +helper/simulate_metal_artifact.m（MATLAB 版伪影合成，对应本项目 simulate_data.py）
- +helper/pkev2kvp.m（对应 util_func.py::pkev2kvp()）
- +helper/interpolate_projection.m（对应 util_func.py::interpolate_projection()）
- +helper/get_mar_params.m（对应 util_func.py::get_mar_params()）
- config/dataset.yaml（数据集配置参数）
- adn/datasets/（Dataset 类定义）

================================================================
六、注意事项
================================================================

1. 不要修改 simulate_data.py 和 util_func.py，这两个文件已经验证正确
2. ODL 库安装可能需要特殊处理：pip install odl
3. DeepLesion 原始数据需要从 NIH 官网下载 Images_png_01~09.zip
4. 所有 .h5 文件使用 gzip 压缩存储
5. 数据归一化范围：线衰减系数 [0.0, 0.5] → [-1, 1]
6. 金属掩模来自 SampleMasks.mat 中的 CT_samples_bwMetal，共 100 个掩模
```

---

以上提示词涵盖了所有退化数据集构建的方法细节。直接复制 ``` 内的内容粘贴给 Cursor 即可。
