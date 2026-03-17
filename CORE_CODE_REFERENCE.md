# 核心代码文件说明

> 本文档列出PnP-ADMM项目中最关键的代码文件和函数

---

## 一、核心算法模块（必看）

### 1. core.py - PnP-ADMM核心算法

**位置**: `/mnt/data/zouhuangbo/cryo-EM/pnp_admm/core.py`

**核心函数**:

```python
def pnp_admm_denoise(y, Hk, Wk, denoiser, T=5, rho=1.0, alpha=1e-3):
    """
    CTF-Constrained PnP-ADMM Denoising
    
    Args:
        y: 观测到的噪声图像 (H, W)
        Hk: CTF在Fourier域 (H, W)
        Wk: 噪声PSD权重 W(k) = 1/(S_n(k) + delta)
        denoiser: 去噪器函数
        T: ADMM迭代次数
        rho: ADMM一致性权重
        alpha: CTF零点稳定性项
    
    Returns:
        dict: {'z_hat': ..., 'x_hat': ..., 'u': ..., 'history': ...}
    """

def weighted_fusion_zstep(Yk, Ck, Wk, rho):
    """z-step A1: 加权融合 (Fourier域闭式解)"""
    Z_tilde = (Wk * Yk + rho * Ck) / (Wk + rho)
    return Z_tilde

def solve_x_step(Zk, Uk, Hk, alpha):
    """x-step: CTF约束投影"""
    Xk = np.conj(Hk) * (Zk + Uk) / (np.abs(Hk)**2 + alpha)
    return x
```

---

### 2. ctf.py - CTF计算

**位置**: `/mnt/data/zouhuangbo/cryo-EM/pnp_admm/ctf.py`

**核心函数**:

```python
def compute_ctf_2d(shape, pixel_size, defocus_u, defocus_v, 
                   defocus_angle, kv, cs, amplitude_contrast=0.1, 
                   phase_flip=True):
    """
    计算2D CTF H(k)
    
    CTF公式:
        λ = 12.264 / sqrt(V * (1 + 0.978478e-6 * V))
        χ = π·λ·defocus·k² - π/2·Cs·λ³·k⁴
        CTF = -√(1-ac²)·sin(χ) - ac·cos(χ)
    """

def load_ctf_params(json_path, mic_idx=None):
    """从JSON加载CTF参数（支持cryocrab格式）"""

def find_ctf_json_for_micrograph(mic_path):
    """自动查找微图对应的CTF JSON文件"""

def load_ctf_for_micrograph(mic_path, shape, ctf_json_path=None):
    """为特定微图加载并计算CTF"""
```

---

### 3. noise_psd.py - 噪声估计

**位置**: `/mnt/data/zouhuangbo/cryo-EM/pnp_admm/noise_psd.py`

**核心函数**:

```python
def compute_noise_psd_from_diff(diff_img):
    """
    从diff图像计算噪声PSD
    
    原理:
        diff = even - odd = n1 - n2 (信号抵消)
        S_d(k) = |FFT(diff)|²
        S_n(k) = (1/2) * S_d(k)
    """

def compute_weight_from_psd(S_n, delta=1e-6, method='exponential'):
    """
    计算逆噪声权重 W(k)
    
    Methods:
        'inverse': W(k) = 1 / (S_n(k) + delta)
        'exponential': W(k) = exp(-S_n / (S_n.mean() + delta)) [推荐]
    """

def load_diff_and_compute_weight(diff_path, delta=None):
    """加载diff文件并计算权重（带自动delta估计）"""
```

---

### 4. denoiser.py - 去噪器接口

**位置**: `/mnt/data/zouhuangbo/cryo-EM/pnp_admm/denoiser.py`

**核心类**:

```python
class TopazDenoiser(Denoiser):
    """Topaz深度去噪器包装器"""
    MODEL_ALIASES = {
        'unet': 'unet_L2_v0.2.2.sav',
        'unet-small': 'unet_small_L1_v0.2.2.sav',
        'fcnn': 'fcnn_L1_v0.2.2.sav',
    }

class GaussianDenoiser(Denoiser):
    """高斯平滑去噪器（测试用）"""
    
class LowpassDenoiser(Denoiser):
    """Butterworth低通滤波器"""

def create_denoiser(name, **kwargs):
    """工厂函数：通过名称创建去噪器"""
```

---

### 5. mrc_io.py - MRC文件读写

**位置**: `/mnt/data/zouhuangbo/cryo-EM/pnp_admm/mrc_io.py`

**核心函数**:

```python
def read_mrc(path):
    """读取MRC文件，返回(image, header_info)"""

def write_mrc(path, data, pixel_size=1.0):
    """写入MRC文件"""
```

---

## 二、主程序与工具

### 6. main.py - 命令行入口

**位置**: `/mnt/data/zouhuangbo/cryo-EM/pnp_admm/main.py`

**使用示例**:
```bash
python main.py --full full.mrc --diff diff.mrc --denoiser unet --cuda
```

**核心流程**:
1. 配对full和diff文件
2. 读取full图像
3. 计算噪声权重W(k)
4. 计算CTF H(k)
5. 创建去噪器
6. 运行PnP-ADMM
7. 保存结果

---

### 7. visualize.py - 可视化

**位置**: `/mnt/data/zouhuangbo/cryo-EM/pnp_admm/visualize.py`

**使用示例**:
```bash
python visualize.py -i full.mrc -z full_z_hat.mrc -d diff.mrc --show-weight
```

---

## 三、关键实验脚本

### 8. 合成数据验证实验

**位置**: `/mnt/data/zouhuangbo/cryo-EM/pnp_admm/experiments/synthetic_ctf_aware/`

| 文件 | 功能 |
|------|------|
| `generate_data.py` | 生成合成cryo-EM数据 |
| `ideal_denoiser.py` | 理想Hx-aware去噪器 |
| `run_experiment.py` | 主实验脚本 |
| `run_iteration_comparison.py` | 迭代次数对比实验 |
| `analyze_results.py` | 结果分析 |

**理想去噪器核心代码**:
```python
class IdealHxDenoiser:
    """理想Hx-aware去噪器"""
    def __init__(self, hx_ground_truth, blend_weight=0.5):
        self.hx_truth = hx_ground_truth
        self.w = blend_weight
    
    def __call__(self, z_tilde):
        return (1 - self.w) * z_tilde + self.w * self.hx_truth
```

---

### 9. 参数扫描实验

**位置**: `/mnt/data/zouhuangbo/cryo-EM/pnp_admm/experiments/`

| 文件 | 功能 |
|------|------|
| `exp_a_rho_scan_cropped.py` | ρ参数扫描 |
| `exp_b_alpha_scan_cropped.py` | α参数扫描 |
| `exp_c_T_scan_cropped.py` | 迭代次数扫描 |
| `exp_d_denoiser_comparison.py` | 去噪器对比 |
| `exp_multi_denoiser_T_comparison_512.py` | 多去噪器T扫描 |

---

### 10. 分析模块

**位置**: `/mnt/data/zouhuangbo/cryo-EM/pnp_admm/experiments/analysis/`

| 文件 | 功能 |
|------|------|
| `metrics.py` | 基础评估指标（PSNR, SSIM, CTF correlation等） |
| `even_odd_metrics.py` | Even/Odd一致性评估（核心物理指标） |
| `plot_convergence.py` | 收敛曲线可视化 |

---

## 四、代码调用流程图

```
main.py / run_experiment.py
    │
    ├──► ctf.py: load_ctf_for_micrograph()
    │      └── 读取JSON → compute_ctf_2d() → Hk
    │
    ├──► noise_psd.py: load_diff_and_compute_weight()
    │      └── read_mrc(diff) → compute_noise_psd_from_diff() → Wk
    │
    ├──► denoiser.py: create_denoiser()
    │      └── TopazDenoiser / GaussianDenoiser / LowpassDenoiser
    │
    └──► core.py: pnp_admm_denoise(y, Hk, Wk, denoiser, T, rho, alpha)
           │
           ├──► weighted_fusion_zstep() [Step A1]
           ├──► denoiser() [Step A2]
           ├──► solve_x_step() [Step B]
           └──► u-update [Step C]
```

---

## 五、关键数学公式速查

### CTF公式
```
λ = 12.264 / sqrt(V * (1 + 0.978478e-6 * V))  [电子波长，Angstrom]
χ = π·λ·defocus·k² - π/2·Cs·λ³·k⁴
CTF = -√(1-ac²)·sin(χ) - ac·cos(χ)
```

### 噪声PSD
```
S_n(k) = (1/2) * |FFT(diff)|²
W(k) = 1 / (S_n(k) + δ)
```

### PnP-ADMM迭代
```
# Step A1: 加权融合
Z̃(k) = (W(k)·Y(k) + ρ·C(k)) / (W(k) + ρ)

# Step A2: 去噪
z = D(z̃)

# Step B: CTF约束投影
X(k) = H*(k)·(Z(k) + U(k)) / (|H(k)|² + α)

# Step C: 对偶更新
u = u + (z - Hx)
```

---

**提示**: 阅读代码时，建议从 `core.py` 的 `pnp_admm_denoise()` 函数开始，这是整个算法的核心入口。
