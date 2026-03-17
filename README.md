# CTF-Constrained PnP-ADMM Denoising - 核心代码包

> **版本**: 1.0  
> **日期**: 2026-03-17  
> **说明**: 本包包含CTF约束PnP-ADMM去噪算法的核心实现和实验报告

---

## 📦 包内容说明

### 1. 核心代码文件（8个）

| 文件 | 行数 | 功能描述 |
|------|------|----------|
| `core.py` | ~236 | **PnP-ADMM核心算法** - 实现z-step、x-step、u-step迭代 |
| `ctf.py` | ~370 | **CTF计算** - CTF公式实现、JSON参数解析 |
| `denoiser.py` | ~351 | **去噪器接口** - Topaz/高斯/低通去噪器包装 |
| `noise_psd.py` | ~237 | **噪声估计** - 从even/odd diff估计噪声PSD |
| `mrc_io.py` | ~68 | **MRC文件IO** - 读写MRC显微图像文件 |
| `main.py` | ~321 | **命令行入口** - 完整的去噪处理流程 |
| `visualize.py` | ~315 | **可视化工具** - 结果对比可视化 |
| `__init__.py` | ~77 | 包初始化，导出主要接口 |

### 2. 实验报告文档（2个）

| 文件 | 说明 |
|------|------|
| `EXPERIMENT_REPORT.md` | **完整实验报告** - 包含算法描述、实验结果、核心发现 |
| `CORE_CODE_REFERENCE.md` | **代码参考文档** - 核心函数说明、调用流程、数学公式 |

### 3. 依赖文件

| 文件 | 说明 |
|------|------|
| `requirements.txt` | Python依赖列表 |

---

## 🚀 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

**基础依赖**: numpy, scipy, mrcfile, matplotlib

**可选依赖** (使用Topaz深度去噪器时需要):
- PyTorch
- Topaz (conda环境)

### 基础使用

```python
from core import pnp_admm_denoise
from ctf import load_ctf_for_micrograph
from noise_psd import load_diff_and_compute_weight
from denoiser import create_denoiser
from mrc_io import read_mrc, write_mrc

# 1. 读取图像
y, header = read_mrc('micrograph_full.mrc')

# 2. 计算CTF H(k)
Hk, ctf_info = load_ctf_for_micrograph('micrograph_full.mrc', y.shape)

# 3. 计算噪声权重W(k)
Wk, weight_info = load_diff_and_compute_weight('micrograph_diff.mrc')

# 4. 创建去噪器
denoiser = create_denoiser('gaussian', sigma=1.0)

# 5. 运行PnP-ADMM
result = pnp_admm_denoise(
    y=y, Hk=Hk, Wk=Wk, denoiser=denoiser,
    T=10, rho=1.0, alpha=1e-3
)

# 6. 保存结果
write_mrc('denoised.mrc', result['z_hat'], pixel_size=header['pixel_size'])
```

### 命令行使用

```bash
# 基础去噪
python main.py --full full.mrc --diff diff.mrc --denoiser gaussian

# 使用Topaz UNet (需要conda环境)
python main.py --full full.mrc --diff diff.mrc --denoiser unet --cuda

# 完整参数
python main.py --full full.mrc --diff diff.mrc \
    --denoiser unet -T 10 --rho 2.0 --alpha 1e-4 \
    --output-z denoised.mrc --save-meta
```

### 可视化结果

```bash
python visualize.py --original full.mrc --denoised denoised.mrc \
    --diff diff.mrc --show-weight --output comparison.png
```

---

## 📊 核心实验结论（详见EXPERIMENT_REPORT.md）

### 主要发现

| 发现 | 内容 |
|------|------|
| **框架有效性** | 使用理想Hx-aware去噪器时，PSNR提升20-50 dB |
| **先验不匹配** | 传统去噪器学习x的分布，但目标是Hx，导致~20 dB性能损失 |
| **迭代重要性** | T=10比T=1好47 dB，迭代次数至关重要 |
| **参数推荐** | ρ=1.0-10.0, α=1e-5-1e-3, T=5-20 |

### 去噪器性能排名

| 排名 | 去噪器 | E_out/E_in Ratio | 说明 |
|-----|--------|------------------|------|
| 1 | Topaz UNet | 0.000056 | 最佳，但最慢(44.9s) |
| 2 | Gaussian σ=2.0 | 0.000058 | 次佳，可能过度平滑 |
| 3 | Gaussian σ=1.0 | 0.00036 | 良好 |
| 4 | Lowpass 0.3 | 0.0027 | 平衡选择 |
| 5 | Identity | 0.048 | 基线（无去噪） |

---

## 📁 代码结构

```
pnp_admm_package/
├── 核心算法模块
│   ├── core.py              # PnP-ADMM迭代算法
│   ├── ctf.py               # CTF计算与JSON解析
│   ├── denoiser.py          # 去噪器接口
│   └── noise_psd.py         # 噪声PSD估计
│
├── 工具模块
│   ├── mrc_io.py            # MRC文件读写
│   ├── main.py              # 命令行入口
│   └── visualize.py         # 可视化工具
│
├── 文档
│   ├── EXPERIMENT_REPORT.md      # 完整实验报告
│   ├── CORE_CODE_REFERENCE.md    # 代码参考文档
│   └── README.md                 # 本文件
│
└── requirements.txt         # Python依赖
```

---

## 🧬 算法原理

### 观测模型
```
y = z + n,  其中 z = Hx
```
- y: 观测到的噪声图像 (full = even + odd)
- H: CTF (Contrast Transfer Function) 算子
- x: 潜在的无CTF投影
- z: 测量域的CTF调制干净图像（目标输出）
- n: 噪声

### PnP-ADMM迭代

```
初始化: z⁰ = y, x⁰ = 0, u⁰ = 0

迭代 t=0..T-1:
    # Step A1: 加权融合
    Z̃(k) = (W(k)·Y(k) + ρ·C(k)) / (W(k) + ρ)
    
    # Step A2: Plug-and-Play去噪
    z = D(z̃)
    
    # Step B: CTF约束投影
    X(k) = H*(k)·(Z(k) + U(k)) / (|H(k)|² + α)
    
    # Step C: 对偶更新
    u = u + (z - Hx)
```

### 噪声PSD估计
```
diff = even - odd = n₁ - n₂
S_n(k) = (1/2) · |FFT(diff)|²
W(k) = 1 / (S_n(k) + δ)
```

---

## 📖 推荐阅读顺序

1. **先读文档**: `EXPERIMENT_REPORT.md` - 了解算法原理和实验结果
2. **再读代码**: `core.py` - 理解PnP-ADMM核心实现
3. **参考文档**: `CORE_CODE_REFERENCE.md` - 查阅具体函数说明
4. **动手实践**: 运行 `main.py` 进行去噪测试

---

## 🔧 参数说明

| 参数 | 默认值 | 说明 | 推荐范围 |
|------|--------|------|----------|
| T | 5 | ADMM迭代次数 | 5-20 |
| ρ | 1.0 | ADMM一致性权重 | 1.0-10.0 |
| α | 1e-3 | CTF零点稳定性项 | 1e-5-1e-3 |
| delta | auto | 噪声权重地板值 | 自动估计 |

---

## ⚠️ 注意事项

1. **Topaz去噪器需要conda环境**: 
   ```bash
   conda activate topaz
   python main.py --denoiser unet --cuda
   ```

2. **内存要求**: 处理大图像(4096×4096)时需要较大内存，建议使用patch模式:
   ```bash
   python main.py --full large.mrc --patch-size 1024 --padding 256
   ```

3. **CTF JSON文件**: 程序会自动查找与微图对应的CTF参数文件，确保文件结构符合规范:
   ```
   dataset/
   ├── micrograph/000000_dataset_full.mrc
   └── CTF/dataset.json
   ```

---

## 📚 相关文档

- `EXPERIMENT_REPORT.md` - 完整实验报告，包含：
  - 算法详细描述
  - ρ/α/T参数扫描结果
  - 去噪器对比实验
  - 合成数据验证（核心发现：~20 dB差距）
  - 核心结论和下一步建议

- `CORE_CODE_REFERENCE.md` - 代码参考文档，包含：
  - 核心函数详细说明
  - 代码调用流程图
  - 关键数学公式速查

---

## 📧 联系方式

如有问题或建议，请参考项目完整代码库或联系项目维护者。

---

**最后更新**: 2026-03-17
