from typing import List, Dict

import numpy as np
import torch
import torch_geometric

import sevenn.util
import sevenn._keys as KEY


# TODO: Now, I'm not sure why this class is required
class AtomGraphData(torch_geometric.data.Data):
    """
    Args:
        x (Tensor, optional): atomic numbers with shape :obj:`[num_nodes,
            atomic_numbers]`. (default: :obj:`None`)
        edge_index (LongTensor, optional): Graph connectivity in COO format
            with shape :obj:`[2, num_edges]`. (default: :obj:`None`)
        edge_attr (Tensor, optional): Edge feature matrix with shape
            :obj:`[num_edges, num_edge_features]`. (default: :obj:`None`)
        y_energy: scalar # unit of eV (VASP raw)
        y_force: [num_nodes, 3] # unit of eV/A (VASP raw)
        y_stress: [6]  # [xx, yy, zz, xy, yz, zx] # unit of eV/A^3 (VASP raw)
        pos (Tensor, optional): Node position matrix with shape
            :obj:`[num_nodes, num_dimensions]`. (default: :obj:`None`)
        **kwargs (optional): Additional attributes.

    x, y_force, pos should be aligned with each other.
    """
    def __init__(
        self,
        x=None,
        edge_index=None,
        pos=None,
        edge_attr=None,
        **kwargs
    ):
        super(AtomGraphData, self).__init__(
            x,
            edge_index,
            edge_attr,
            pos=pos
        )

        # This is very strange... but I couldn't find a better way yet
        self[KEY.NODE_ATTR] = x

        for k, v in kwargs.items():
            self[k] = v

    #def to_dict(self):
    #    """
    #    'maybe' Dict[str, Tensor]
    #    """
    #    return {k: v for k, v in self.items()}

    def to_numpy_dict(self):
        # This is not debuged yet!
        dct = {k: v.detach().cpu().numpy() if type(v) is torch.Tensor else v
               for k, v in self.items()}
        return dct

    def fit_dimension(self):
        natoms = self.num_atoms.item()
        for k, v in self.items():
            if not isinstance(v, torch.Tensor):
                continue
            if natoms == 1 and (k == KEY.FORCE or k == KEY.ATOMIC_ENERGY):
                continue
            self[k] = v.squeeze()
        return self

    @staticmethod
    def from_numpy_dict(dct):
        for k, v in dct.items():
            if k == KEY.CELL_SHIFT:
                dct[k] = torch.Tensor(v)  # this is special
            else:
                dct[k] = sevenn.util.dtype_correct(v)
        return AtomGraphData(**dct)

