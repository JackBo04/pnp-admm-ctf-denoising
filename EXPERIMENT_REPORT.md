# CTF-Constrained PnP-ADMM 去噪实验报告

> **项目**: 冷冻电镜图像CTF约束PnP-ADMM去噪框架  
> **日期**: 2026年2月  
> **目标**: 验证PnP-ADMM框架在CTF-aware去噪场景下的有效性

---

## 1. 核心算法概述

### 1.1 问题定义

**观测模型**:
```
y = z + n,  其中 z = Hx
```

- **y**: 观测到的噪声图像 (full = even + odd)
- **H**: CTF (Contrast Transfer Function) 算子
- **x**: 潜在的无CTF投影 (latent space)
- **z**: 测量域的CTF调制干净图像 (主要输出)
- **n**: 噪声

**核心目标**: 在测量域进行去噪，输出保持CTF调制的图像 ẑ ≈ Hx（不是CTF-free的x）

### 1.2 PnP-ADMM迭代算法

```python
# 初始化
z⁰ = y, x⁰ = 0, u⁰ = 0

# 迭代 t=0..T-1
for t in range(T):
    # Step A1: 加权融合 (Fourier域闭式解)
    cᵗ = Hxᵗ - uᵗ
    Z̃(k) = (W(k)·Y(k) + ρ·Cᵗ(k)) / (W(k) + ρ)
    
    # Step A2: Plug-and-Play去噪器
    zᵗ⁺¹ = D(z̃)
    
    # Step B: x-step (CTF约束投影)
    X(k) = H*(k)·(Z(k) + U(k)) / (|H(k)|² + α)
    
    # Step C: u-step (对偶更新)
    uᵗ⁺¹ = uᵗ + (zᵗ⁺¹ - Hxᵗ⁺¹)
```

### 1.3 噪声PSD估计

利用odd/even diff文件估计噪声功率谱密度：
```
diff = even - odd = n₁ - n₂  (信号抵消)
S_d(k) = E[|D(k)|²] = 2·S_n(k)
S_n(k) = (1/2) · S_d(k)
W(k) = 1 / (S_n(k) + δ)  (逆噪声加权)
```

---

## 2. 实验设计与方法

### 2.1 实验体系

| 实验类型 | 目的 | 数据集 |
|---------|------|--------|
| **参数扫描** | 确定最优ρ, α, T | 真实cryo-EM显微图像 |
| **去噪器对比** | 评估不同先验的效果 | 真实cryo-EM显微图像 |
| **合成数据验证** | 验证框架本身有效性 | 合成256×256粒子图像 |

### 2.2 评估指标

**1. Even/Odd一致性（核心物理指标）**
```
E_in = ||even - odd||² = ||diff||²
E_out = ||z_even - z_odd||²
ratio = E_out / E_in  (应显著下降，< 1表示有改善)
improvement = (1 - ratio) × 100%
```

**2. Primal Residual（CTF一致性）**
```
r = ||z - Hx||
```
- 应随迭代稳定下降
- 反映CTF约束的满足程度

**3. Signal Retention（信号保留率）**
```
signal_retention = ||z||² / ||y||² × 100%
```
- 衡量去噪过程中信号保留的程度
- 过低表示过度平滑

---

## 3. 真实数据实验结果

### 3.1 ρ参数扫描实验

**实验设置**:
- 去噪器: Lowpass (cutoff=0.05)
- 固定参数: T=10, α=0.001
- 图像尺寸: 1024×1024
- 测试ρ值: 0.1, 0.5, 1.0, 2.0, 5.0, 10.0

**关键结果**:

| ρ | E_out/E_in Ratio | Improvement % | Final Residual | 收敛速度 |
|---|------------------|---------------|----------------|----------|
| 0.1 | 0.00124 | 99.88% | 0.65 | 0.690 |
| 0.5 | 0.000096 | 99.99% | 0.32 | 0.663 |
| **1.0** | **0.000029** | **99.997%** | **0.23** | **0.662** |
| 2.0 | 0.0000083 | 99.999% | 0.16 | 0.636 |
| 5.0 | 0.0000015 | 99.9998% | 0.13 | 0.479 |
| 10.0 | 0.0000004 | 99.99996% | 0.11 | 0.329 |

**结论**: 
- ρ越大，E_out/E_in ratio越小，收敛越好
- ρ=10时达到99.99996%的改善
- 但ρ过大可能导致过度平滑，需要在ratio和signal retention间平衡

![rho扫描结果](./experiments/results/exp_a_rho_scan/rho_scan_summary.png)

### 3.2 去噪器对比实验

**实验设置**:
- 固定参数: T=10, ρ=10.0, α=0.001
- 对比去噪器: Identity, Gaussian(σ=1.0, 2.0), Lowpass(cutoff=0.3, 0.5), Topaz UNet

**关键结果**:

| 去噪器 | E_out/E_in Ratio | Improvement % | Final Residual | 时间(秒) |
|--------|------------------|---------------|----------------|----------|
| Identity | 0.048 | 95.23% | 3.27 | 8.77 |
| Gaussian σ=1.0 | 0.00036 | 99.96% | 2.80 | 8.70 |
| **Gaussian σ=2.0** | **0.000058** | **99.994%** | **0.90** | **8.80** |
| Lowpass 0.3 | 0.0027 | 99.73% | 2.37 | 9.91 |
| Lowpass 0.5 | 0.012 | 98.81% | 2.14 | 9.95 |
| **Topaz UNet** | **0.000056** | **99.994%** | **1.82** | **44.92** |

**结论**:
- **Topaz UNet** 和 **Gaussian σ=2.0** 表现最好（ratio < 0.00006）
- Topaz UNet 最慢（44.92秒 vs 8-10秒）
- Identity 最差（ratio=0.048），说明去噪器先验的重要性

![去噪器对比](./experiments/results/exp_d_denoiser_comparison/denoiser_comparison.png)

### 3.3 多去噪器T扫描实验

**实验设置**:
- 固定参数: ρ=0.1, α=0.001
- T值: 5, 10, 15
- 去噪器: Identity, Gaussian(1.0, 2.0), Lowpass(0.3), Topaz UNet

**关键结果**:

| 去噪器 | T | Ratio | Improvement % | Signal Retention % |
|--------|---|-------|---------------|-------------------|
| Identity | 5 | 0.393 | 60.65% | 74.41% |
| Identity | 10 | 0.435 | 56.50% | 78.71% |
| Identity | 15 | 0.456 | 54.39% | 80.62% |
| Gaussian 1.0 | 5 | 0.041 | 95.88% | 38.99% |
| Gaussian 2.0 | 5 | 0.007 | 99.30% | 25.22% |
| Lowpass 0.3 | 5 | 0.141 | 85.90% | 56.02% |
| **Topaz UNet** | **5** | **0.00078** | **99.92%** | **9.92%** |
| Topaz UNet | 10 | 0.0017 | 99.83% | 17.47% |
| Topaz UNet | 15 | 0.0033 | 99.67% | 25.46% |

**关键发现**:
- **Trade-off**: 更好的ratio通常意味着更低的signal retention
- Topaz UNet T=5时ratio最佳(0.00078)，但signal retention最低(9.92%)
- Lowpass 0.3在ratio和signal retention间取得较好平衡

![多去噪器对比](./experiments/results/multi_denoiser_T_512_left/metrics_comparison.png)

---

## 4. 合成数据验证实验（核心发现）

### 4.1 实验动机

**核心矛盾**: Topaz/高斯/低通滤波等去噪器学习的是**干净图像x**的先验，但PnP-ADMM的目标是恢复**CTF调制后的图像Hx**。存在**先验不匹配**问题。

**验证思路**: 在合成数据上，用"理想"的Hx-aware去噪器（利用地面真值插值）验证框架本身是否正确。

### 4.2 合成数据生成流程

```
Step 1: 生成干净粒子投影 x
    └─ 使用高斯blob模拟蛋白质结构
           
Step 2: CTF调制
    └─ Hx = IFFT{FFT(x) × CTF(k)}
    └─ 固定CTF参数: 300kV, Cs=2.7mm, 欠焦1.5μm
           
Step 3: 添加高斯噪声
    └─ even = Hx + n₁
    └─ odd  = Hx + n₂
           
Step 4: 计算full/diff
    └─ full = even + odd
    └─ diff = even - odd (纯噪声，用于估计W(k))
```

**数据规格**:
- 图像尺寸: 256×256像素
- 像素大小: 1.0 Å
- 噪声水平: σ = 0.5

### 4.3 "理想"去噪器设计

利用地面真值Hx和输入z_tilde做插值：

```python
class IdealHxDenoiser:
    """理想Hx-aware去噪器"""
    def __init__(self, hx_ground_truth, blend_weight=0.5):
        self.hx_truth = hx_ground_truth
        self.w = blend_weight
    
    def __call__(self, z_tilde):
        return (1 - self.w) * z_tilde + self.w * self.hx_truth
```

测试的变体:
- **identity**: w=0.0 (基线，无去噪)
- **ideal_w03**: w=0.3 (弱去噪)
- **ideal_w05**: w=0.5 (中等去噪)
- **ideal_w07**: w=0.7 (强去噪)
- **oracle**: w=1.0 (理论上限)

### 4.4 合成实验关键结果

#### 结果1: 理想去噪器显著优于传统方法

| 去噪器 | PSNR (dB) | MSE | 结论 |
|--------|-----------|-----|------|
| identity | 1.94 | 1.0 | 基线 |
| gaussian | 1.94 | 1.0 | ❌ 无效 |
| lowpass | 1.94 | 1.0 | ❌ 无效 |
| **ideal_w05** | **19.95** | **0.016** | ✅ **+18 dB** |
| **ideal_w07** | **33.16** | **0.00075** | ✅ **+31 dB** |
| oracle | 101.94 | ~0 | 理论上限 |

**平均差距**: 理想方法(21.4 dB) vs 传统方法(1.9 dB) = **~20 dB差距**

#### 结果2: 迭代次数至关重要

使用ideal_w05去噪器，不同迭代次数的结果：

| T (迭代) | PSNR (dB) | CTF Error | 相对于T=1提升 |
|---------|-----------|-----------|--------------|
| 1 | 7.96 | 0.128 | - |
| 3 | 19.95 | 0.032 | +12.0 dB |
| 5 | 31.78 | 0.008 | +23.8 dB |
| 10 | **55.12** | **0.00025** | **+47.2 dB** |

**结论**: 迭代次数越多，CTF一致性误差指数级下降，PSNR显著提升。

#### 结果3: x-step有效保持CTF一致性

所有方法的CTF一致性指标都很好：
- **原因**: x-step强制执行x = argmin ||Hx - (z+u)||²
- **结果**: 即使传统去噪器，||z - Hx||也很小

**但关键区别**:
- CTF一致性好 ≠ 去噪质量好
- 传统方法虽然满足CTF约束，但偏离真值（因为先验不匹配）
- 理想方法既满足CTF约束，又接近真值

### 4.5 合成实验可视化结果

**各方法对比网格**:

![合成实验对比网格](./experiments/synthetic_ctf_aware/synthetic_analysis/comparison_grid.png)

**方法对比总结**:

![合成实验总结](./experiments/synthetic_ctf_aware/synthetic_analysis/summary_comparison.png)

**迭代次数分析**:

![迭代对比](./experiments/synthetic_ctf_aware/synthetic_analysis/iteration_comparison/iteration_comparison_fixed.png)

---

## 5. 核心结论

### 5.1 主要发现

1. **框架本身是正确的**: 当使用理想Hx-aware去噪器时，PnP-ADMM能够收敛并得到很好的结果（PSNR提升20-50 dB）。

2. **问题在于去噪器先验不匹配**: 传统去噪器（Topaz/高斯/低通）学习的是x的分布，而我们需要恢复的是Hx。

3. **x-step有效保持CTF一致性**: 所有方法的CTF一致性都很好，说明CTF约束投影步骤工作正常。

4. **迭代次数很重要**: 更多迭代带来更好的结果（T=10比T=1好47 dB）。

### 5.2 参数推荐

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| ρ | 1.0-10.0 | 越大收敛越快，但可能过度平滑 |
| α | 1e-5 - 1e-3 | 越小residual越小 |
| T | 5-20 | 越大效果越好，但时间增加 |

### 5.3 去噪器选择建议

| 场景 | 推荐去噪器 | 理由 |
|------|-----------|------|
| 追求最佳ratio | Topaz UNet (T=5) | ratio=0.00078，但signal retention低 |
| 平衡选择 | Lowpass 0.3 | ratio=0.14，signal retention=56% |
| 快速测试 | Gaussian 2.0 | ratio=0.007，速度快 |
| 验证框架 | Identity | 基线对比 |

---

## 6. 核心代码文件

### 6.1 算法实现（5个核心文件）

| 文件 | 功能 | 关键类/函数 |
|------|------|------------|
| **core.py** | PnP-ADMM核心算法 | `pnp_admm_denoise()`, `solve_x_step()`, `weighted_fusion_zstep()` |
| **ctf.py** | CTF计算与参数解析 | `compute_ctf_2d()`, `load_ctf_params()`, `find_ctf_json_for_micrograph()` |
| **denoiser.py** | 去噪器接口 | `TopazDenoiser`, `GaussianDenoiser`, `LowpassDenoiser`, `create_denoiser()` |
| **noise_psd.py** | 噪声PSD估计 | `compute_noise_psd_from_diff()`, `compute_weight_from_psd()` |
| **mrc_io.py** | MRC文件读写 | `read_mrc()`, `write_mrc()` |

### 6.2 主程序与可视化

| 文件 | 功能 |
|------|------|
| **main.py** | 命令行入口，完整处理流程 |
| **visualize.py** | 结果可视化对比 |
| **run_in_topaz_env.sh** | Conda环境运行脚本 |

### 6.3 实验脚本（experiments/目录）

| 文件 | 功能 |
|------|------|
| **exp_a_rho_scan_cropped.py** | ρ参数扫描实验 |
| **exp_b_alpha_scan_cropped.py** | α参数扫描实验 |
| **exp_c_T_scan_cropped.py** | 迭代次数扫描实验 |
| **exp_d_denoiser_comparison.py** | 去噪器对比实验 |
| **exp_multi_denoiser_T_comparison_512.py** | 多去噪器T扫描 |
| **synthetic_ctf_aware/run_experiment.py** | 合成数据验证实验 |

---

## 7. 下一步工作建议

### 7.1 短期目标

1. **训练CTF-conditioned去噪器**
   ```python
   # 输入: [含噪图像y, CTF(k)]
   # 输出: 去噪后的Hx
   model = UNet(in_channels=2, out_channels=1)
   ```

2. **优化参数组合**: 针对特定去噪器寻找最优ρ, α, T

3. **更大规模测试**: 在更多真实显微图像上验证

### 7.2 长期目标

1. **端到端学习**: 联合优化去噪器和ADMM参数
2. **多CTF鲁棒性**: 训练对CTF参数变化鲁棒的去噪器
3. **计算效率**: 优化Topaz等深度学习去噪器的速度

---

## 附录A: 快速开始

```bash
# 1. 基础去噪（无需conda环境）
python main.py --full full.mrc --diff diff.mrc --denoiser gaussian

# 2. 使用Topaz UNet（需要conda环境）
./run_in_topaz_env.sh --full full.mrc --diff diff.mrc --denoiser unet --cuda

# 3. 完整参数控制
python main.py --full full.mrc --diff diff.mrc \
    --denoiser unet -T 10 --rho 2.0 --alpha 1e-4 \
    --output-z denoised.mrc --save-meta

# 4. 可视化结果
python visualize.py --original full.mrc --denoised full_z_hat.mrc --diff diff.mrc --show-weight
```

## 附录B: 实验数据索引

所有实验结果位于 `experiments/results/` 目录:

- `exp_a_rho_scan/` - ρ扫描结果（含summary.json和可视化图）
- `exp_b_alpha_scan/` - α扫描结果
- `exp_c_T_scan/` - T扫描结果
- `exp_d_denoiser_comparison/` - 去噪器对比
- `multi_denoiser_T_512/` - 多去噪器T扫描
- `synthetic_ctf_aware/synthetic_analysis/` - 合成数据验证结果

---

**文档版本**: 1.0  
**最后更新**: 2026-03-17
