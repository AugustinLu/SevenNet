"""Python side of pair style e3gnn on the Apple GPU (pair_e3gnn_mps/): owns the
calculator and computes for LAMMPS.

pair_coeff * * checkpoint.pth [modal NAME] element1 element2 ...
modal selects the fidelity of a multi-fidelity model.

"""

from typing import List, Sequence

import numpy as np
import torch
from ase import Atoms
from ase.data import atomic_numbers

from sevenn.calculator import SevenNetCalculator

_VIRIAL = [
    0,
    1,
    2,
    5,
    4,
    3,
]  # ASE voigt xx yy zz yz xz xy to LAMMPS xx yy zz xy xz yz


def _cell(box) -> np.ndarray:
    lo, hi, xy, xz, yz, _ = box
    return np.array(
        [
            [hi[0] - lo[0], 0.0, 0.0],
            [xy, hi[1] - lo[1], 0.0],
            [xz, yz, hi[2] - lo[2]],
        ]
    )


def _parse(args: List[str]) -> dict:
    out = dict(modal=None)
    i = 0
    while i < len(args) and args[i] == 'modal':
        if i + 1 == len(args):
            raise ValueError('e3gnn: modal requires a name')
        out['modal'] = args[i + 1]
        i += 2
    out['symbols'] = args[i:]
    bad = [s for s in out['symbols'] if s not in atomic_numbers]
    if bad:
        raise ValueError(
            f'e3gnn: {bad[0]} is neither a keyword (modal) nor an element symbol'
        )
    return out


class PairDriver:
    def __init__(self, args: List[str], ntypes: int, box: Sequence) -> None:
        if not torch.backends.mps.is_available():
            raise RuntimeError(
                'e3gnn: this pair style needs the Apple GPU, and MPS is not '
                'available'
            )
        path, opt = args[0], _parse(args[1:])
        if path.endswith('.pt'):
            raise ValueError(
                f'e3gnn: {path} looks like a deployed model. TorchScript cannot '
                f'hold the Metal kernels: give the .pth checkpoint or a pretrained '
                f'name'
            )
        if len(opt['symbols']) < ntypes:
            raise ValueError(
                f'e3gnn: not enough elements in pair_coeff: {ntypes} atom types, '
                f'got {opt["symbols"]}'
            )
        self.calc = SevenNetCalculator(path, device='mps', modal=opt['modal'])
        known = set(self.calc.type_map)
        unknown = [s for s in opt['symbols'] if atomic_numbers.get(s) not in known]
        if unknown:
            raise ValueError(f'e3gnn: elements unknown to this model: {unknown}')
        self.numbers = np.array([0] + [atomic_numbers[s] for s in opt['symbols']])
        self.cutoff = float(self.calc.cutoff)
        rt = self.calc.mps_runtime
        fused = rt is not None and rt.mode == 'fused'
        run = rt.report if fused else 'reference code'
        self.banner = (
            f'e3gnn on the Apple GPU: {path}, cutoff {self.cutoff} A, '
            f'elements {" ".join(opt["symbols"])}\n{run}'
        )

    def compute(self, nlocal: int, x, f, types, eatom, virial, box) -> float:
        if (
            nlocal == 0
        ):  # all atoms lost during a run, with thermo_modify lost ignore
            return 0.0
        lo, _, _, _, _, periodic = box
        forces = np.frombuffer(f, dtype=np.float64).reshape(nlocal, 3)
        out = np.frombuffer(virial, dtype=np.float64)
        atoms = Atoms(
            numbers=self.numbers[np.frombuffer(types, dtype=np.int32)],
            positions=np.frombuffer(x, dtype=np.float64).reshape(nlocal, 3)
            - np.array(lo),
            cell=_cell(box),
            pbc=[bool(p) for p in periodic],
        )
        atoms.calc = self.calc
        energy = atoms.get_potential_energy()
        forces += atoms.get_forces()
        out[:] = -(atoms.get_stress(voigt=True) * atoms.get_volume())[_VIRIAL]
        if eatom is not None:
            np.frombuffer(eatom, dtype=np.float64)[:] += atoms.calc.results[
                'energies'
            ]
        return float(energy)
