# Accelerators

This document describes available accelerator integrations in SevenNet and their installation guide.

:::{caution}
We do not support CuEquivariance and LunaTP for [LAMMPS: Torch](./lammps_torch.md). You must use [LAMMPS: ML-IAP](./lammps_mliap.md) for CuEquivariance and LunaTP.
:::

[FlashTP](https://openreview.net/forum?id=wiQe95BPaB), [CuEquivariance](https://github.com/NVIDIA/cuEquivariance), and [OpenEquivariance](https://github.com/PASSIONLab/OpenEquivariance) provide acceleration for both SevenNet training and inference.

:::{tip}
For small systems, FlashTP with [LAMMPS: Torch](./lammps_torch.md) shows performance advantage over cuEquivariance with [LAMMPS: ML-IAP](./lammps_mliap.md).
A performance crossover occurs at around 10³ atoms, beyond which cuEquivariance becomes more efficient.  (For more information, check section 2.7 of the [SevenNet-Omni paper](https://doi.org/10.1038/s41467-026-70195-8))

FlashTP with [LAMMPS: Torch](./lammps_mliap.md) is generally faster than FlashTP with [LAMMPS: ML-IAP](./lammps_mliap.md).
:::

## [CuEquivariance](https://github.com/NVIDIA/cuEquivariance)

CuEquivariance is an NVIDIA Python library designed to facilitate the construction of high-performance geometric neural networks using segmented polynomials and triangular operations. CuEquivariance accelerates SevenNet during training, inference with ASE and LAMMPS via ML-IAP.

### Requirements
- Python >= 3.10
- cuEquivariance >= 0.6.1

### Installation
After installation of SevenNet, install cuEquivariance via the pip extras:

```bash
pip install sevenn[cueq12]  # For CUDA 12.x
pip install sevenn[cueq13]  # For CUDA 13.x
```

This will install `cuequivariance`, `cuequivariance-torch`, and the corresponding `cuequivariance-ops-torch-cu{12,13}`.

Alternatively, you can install cuEquivariance manually following [their guideline](https://github.com/NVIDIA/cuEquivariance).

:::{note}
Some GeForce GPUs do not support `pynvml`,
causing `pynvml.NVMLError_NotSupported: Not Supported`.
Then try a lower cuEquivariance version, such as 0.6.1.
:::

Check your installation:
```bash
python -c 'from sevenn.nn.cue_helper import is_cue_available; print(is_cue_available())'
True
```

## [FlashTP](https://github.com/SNU-ARC/flashTP)

FlashTP, presented in [FlashTP: Fused, Sparsity-Aware Tensor Product for Machine Learning Interatomic Potentials](https://openreview.net/forum?id=wiQe95BPaB), is a high-performance Tensor-Product library for Machine Learning Interatomic Potentials (MLIPs).

FlashTP accelerates SevenNet during both training and inference, achieving up to ~4× speedup.

### Requirements
- Python >= 3.10
- flashTP >= 0.1.0
- CUDA toolkit >= 12.0

### Installation
Choose `CUDA_ARCH_LIST` for your GPU(s) (see [compute compatibility](https://developer.nvidia.com/cuda/gpus))

```bash
git clone https://github.com/SNU-ARC/flashTP.git
cd flashTP
pip install -r requirements.txt
CUDA_ARCH_LIST="80;90" pip install . --no-build-isolation
```

:::{note}
During installation of FlashTP,
`subprocess.CalledProcessError: ninja ... exit status 137`
typically indicates out-of-memory during compilation.
Try reducing the build parallelism:
```bash
export MAX_JOBS=1
```
:::

Check your installation:
```bash
python -c 'from sevenn.nn.flash_helper import is_flash_available; print(is_flash_available())'
True
```

For more information, see [FlashTP](https://github.com/SNU-ARC/flashTP).

## [OpenEquivariance](https://github.com/PASSIONLab/OpenEquivariance)

OpenEquivariance (oEq) is a library for acceleration of equivariant tensor products. Unlike cuEquivariance, its kernels are compiled once per irreps configuration, eliminating runtime recompilation overhead during training on diverse datasets.

### Requirements
- Python >= 3.10
- openequivariance

### Installation
```bash
pip install sevenn[oeq]
```

Check your installation:
```bash
python -c 'from sevenn.nn.oeq_helper import is_oeq_available; print(is_oeq_available())'
True
```

For more information, see [OpenEquivariance](https://github.com/PASSIONLab/OpenEquivariance).

## [LunaTP (experimental)](https://github.com/HPC-AI-Team/LunaEquivariance)

:::{caution}
LunaTP support in SevenNet is currently experimental.
:::

LunaTP (LunaEquivariance package) replaces the tensor product convolution with a fused kernel that accumulates the messages of each destination atom in registers before one store to global memory. SevenNet also replaces the linear layers, the self-connection tensor product, and the gate nonlinearity with their LunaEquivariance counterparts. The kernels are compiled at the first use of each irreps configuration and cached on disk.

### Requirements
- Python >= 3.10, PyTorch >= 2.8
- NVIDIA GPU with compute capability 8.0 or later, CUDA toolkit with `nvcc`
- LunaEquivariance

### Installation
```bash
pip install sevenn[lunatp]
```

Check your installation:
```bash
python -c 'from sevenn.nn.lunatp_helper import is_lunatp_available; print(is_lunatp_available())'
True
```

### Kernel compilation
The first calculation with a model compiles its LunaTP kernels one after another, which takes several minutes. We recommend `sevenn lunatp_precompile`, which compiles the kernels of a checkpoint in parallel before the first calculation:
```bash
sevenn lunatp_precompile 7net-omni           # kernels for inference (energy, forces, stress)
sevenn lunatp_precompile 7net-omni --train   # also the kernels for training with force and stress losses
```
The compiled kernels are cached on disk, and later calculations load them. Run the command in the environment of the later calculations, since the cache depends on the Python environment, `LD_LIBRARY_PATH`, and the GPU architecture. `--max_workers` sets the number of parallel processes (default: min(8, available CPUs)).

For more information, see [LunaEquivariance](https://github.com/HPC-AI-Team/LunaEquivariance).

## Apple GPU (Metal)

On Macs with Apple silicon, SevenNet runs on the GPU (`device='mps'`) with fused Metal kernels. Each convolution gets one generated kernel that gathers the neighbor features, contracts them with the nonzero Clebsch–Gordan coefficients of the model and sums them into the destination atom, with no per-edge message tensor. One thread owns each output row, so the sums need no atomics and are deterministic. Three more kernels give the first derivatives, and since the product is linear in each of its inputs, every higher derivative is again one of these kernels: training with force and stress losses works with no extra code. Linear, self-connection and gate layers have their own kernels. The edge list is padded to a few fixed sizes, so molecular dynamics keeps reusing the same Metal graphs.

The kernels are generated from the e3nn modules of the model when the calculator is built, and compiled at run time by PyTorch (`torch.mps.compile_shader`). Existing checkpoints run unchanged, and nothing is installed beyond PyTorch.

### Requirements
- A Mac with Apple silicon and macOS with MPS support
- PyTorch >= 2.7, tested on 2.10 to 2.15. On 2.7 to 2.9 a model instance cannot be copied, so `file_type='model_instance'` runs the reference code with a warning; checkpoints use the kernels. With PyTorch 2.6 SevenNet runs the reference e3nn code on the GPU, with a warning; before 2.6 the Apple GPU cannot run SevenNet.
- float32 only: the Apple GPU has no float64. Shift and scale kept in float64 are converted to float32 when the model moves to the GPU.

### Usage
```python
from sevenn.calculator import SevenNetCalculator
calc = SevenNetCalculator(model='7net-0', device='mps')  # fused kernels by default on 'mps'
print(calc.mps_runtime.report)  # which modules run in kernels
```
`enable_mps=False`, or the environment variable `SEVENNET_ENABLE_MPS=0`, runs the reference e3nn code on the same GPU. `enable_mps=True` with `device='auto'` selects the Apple GPU; with another device it is an error. The Apple kernels are never imported on CPU or CUDA. Training and batch inference use the kernels too:
```bash
sevenn train input.yaml --enable_mps
sevenn inference checkpoint_best.pth structures/* --enable_mps
```
For LAMMPS on a Mac, see [LAMMPS: PyTorch](./lammps_torch.md#apple-gpu-serial).

:::{note}
- The kernels are not available in a deployed TorchScript model. The LAMMPS pair style for Apple GPUs therefore loads the checkpoint itself.
- In eval mode, the linear and gate kernels give first derivatives only, and parameter gradients are not computed. Training uses train mode, where every layer is differentiable to any order.
- The kernels index arrays with 32-bit integers, which limits one graph to about 600,000 edges for 7net-omni and 1.9 million for 7net-0. Larger graphs stop with an error that suggests `enable_mps=False`.
:::

The fused gather–product–scatter follows the idea of [FlashTP](https://openreview.net/forum?id=wiQe95BPaB) for NVIDIA GPUs; the Metal kernels are written for Apple GPUs and share no code with it.

## Usage
After the installation, you can leverage the accelerator with appropriate flag (`--enable_cueq`), (`--enable_flash`), (`--enable_oeq`), or (`--enable_lunatp`) options. On Apple GPUs use `device='mps'` or `--enable_mps`.

- [Training](./cli.md#sevenn-train)
- [ASE Calculator](./ase_calculator.md)
- {ref}`LAMMPS<sevenn-get-model>`
