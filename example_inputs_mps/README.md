# Examples for Macs with Apple silicon

These are the examples of `example_inputs/`, set up for the Apple GPU (`device='mps'`, fused Metal kernels).
See the [Apple GPU section](../docs/source/user_guide/accelerator.md#apple-gpu-metal) of the documentation.
There is no parallel MD example: LAMMPS on the Apple GPU runs on one MPI rank.

## Training (`training/`)

The same model and data as `example_inputs/training` (HfO2, read from `example_inputs/data`):
```bash
cd training
sevenn train input_full.yaml --enable_mps -s
sevenn inference checkpoint_best.pth '../../example_inputs/data/label_1/OUTCAR_*' --enable_mps
```

## Molecular dynamics with LAMMPS (`md_serial_example/`)

`pair_coeff` takes a checkpoint (`.pth`) or a pretrained name, not a deployed TorchScript model (`.pt`):
TorchScript cannot hold the Metal kernels, so the pair style runs SevenNet from the checkpoint and there is no
deployment step. `in.lmp` uses `7net-0`; `../training/checkpoint_best.pth` works too.

Build LAMMPS with the Apple GPU pair style (no libtorch link is needed):
```bash
git clone https://github.com/lammps/lammps.git lammps_sevenn --branch stable_22Jul2025_update4 --depth=1
sevenn patch_lammps ./lammps_sevenn --enable_mps
cd lammps_sevenn && mkdir build && cd build
cmake ../cmake -DBUILD_MPI=no -DBUILD_OMP=no && make -j 8
```
then run from `md_serial_example/`:
```bash
/path/to/lammps_sevenn/build/lmp -in in.lmp
```
The log shows `e3gnn on the Apple GPU: 7net-0, ...`.

Or, without building LAMMPS, load the pair style as a plugin into `pip install lammps mpich`:
```bash
sh $(python -c 'import sevenn, os; print(os.path.dirname(sevenn.__file__))')/pair_e3gnn_mps/build_plugin.sh python
export LAMMPS_PLUGIN_PATH=$(python -c 'import sevenn, os; print(os.path.dirname(sevenn.__file__))')/pair_e3gnn_mps
lmp -in in.lmp
```

## ASE

```python
from sevenn.calculator import SevenNetCalculator
calc = SevenNetCalculator('7net-0', device='mps')  # 'auto' would pick the CPU on a Mac
```
