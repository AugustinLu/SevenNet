import os
import warnings
from typing import Optional, Union, List

import numpy as np
import torch
import ase
from e3nn.o3 import Irreps

import sevenn._keys as KEY
import sevenn.util as util
from sevenn.atom_graph_data import AtomGraphData
from sevenn.nn.sequential import AtomGraphSequential
from sevenn.train.dataload import unlabeled_atoms_to_graph


class SevenNetDescriptor:
    """
    Descriptor extraction using SevenNet (E(3)-equivariant GNN).
    This class loads a SevenNet model and runs a forward pass to extract
    intermediate node embeddings (L=0 invariants) from a specified layer.
    """

    def __init__(
        self,
        model: Union[str, AtomGraphSequential] = '7net-0',
        device: Union[torch.device, str] = 'auto',
        layer_index: int = -1,  # -1 means the final convolution layer
        features_dim: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize the SevenNetDescriptor.

        Args:
            model: Name of the pretrained model, path to checkpoint, or an
                   AtomGraphSequential instance.
            device: 'auto', 'cpu', or 'cuda'/'mps'.
            layer_index: The index of the convolution layer to extract features from.
                         -1 means the last convolution layer.
            features_dim: Optional expected feature dimension. If the extracted L=0
                          features do not match this dimension, a warning is raised.
        """
        if isinstance(device, str):
            if device == 'auto':
                self.device = torch.device(
                    'cuda' if torch.cuda.is_available() else 'cpu')
            else:
                self.device = torch.device(device)
        else:
            self.device = device

        self.layer_index = layer_index
        self.features_dim = features_dim

        # Load model
        if isinstance(model, str):
            cp = util.load_checkpoint(model)
            enable_cueq = os.getenv('SEVENNET_ENABLE_CUEQ') == '1'
            enable_flash = os.getenv('SEVENNET_ENABLE_FLASH') == '1'
            enable_oeq = os.getenv('SEVENNET_ENABLE_OEQ') == '1'

            self.model = cp.build_model(
                enable_cueq=enable_cueq,
                enable_flash=enable_flash,
                enable_oeq=enable_oeq)
            self.cutoff = cp.config[KEY.CUTOFF]
        elif isinstance(model, AtomGraphSequential):
            self.model = model
            if self.model.cutoff == 0.0:
                raise ValueError('Model cutoff seems not initialized')
            self.cutoff = self.model.cutoff
        else:
            raise ValueError(
                'model must be a path/name or AtomGraphSequential instance.')

        self.model.set_is_batch_data(False)
        self.model.eval_type_map = True
        self.model.to(self.device)
        self.model.eval()

        # Identify convolution layers in the model
        self.conv_modules = []
        for name, module in self.model.named_children():
            if 'self_interaction_2' in name or name.endswith('equivariant_gate'):
                # We can either extract from the gate output or self_interaction_2 output
                # The output of self_interaction_2 or gate is the updated node
                # feature
                pass
            if 'self_interaction_1' in name:
                # To be precise, let's just collect layers and slice exactly where we
                # need.
                pass

        # Better strategy: the layer blocks follow a pattern:
        # e.g., 0_self_connection_intro, 0_self_interaction_1, 0_convolution, 0_self_interaction_2, 0_self_connection_outro, 0_equivariant_gate
        # We want the output after the equivariant gate (which is the final state of the layer's node features)
        # or after `self_connection_outro` if gate doesn't exist.
        self._target_module_name = self._find_target_module(layer_index)

    def _find_target_module(self, layer_index: int) -> str:
        """Finds the name of the module from which to extract the output."""
        # Find all modules that represent the end of a message passing block
        # Usually, this is the equivariant gate, e.g., '0_equivariant_gate', '1_equivariant_gate'
        # Let's collect them
        gate_modules = []
        for name, module in self.model.named_children():
            if name.endswith('equivariant_gate'):
                gate_modules.append(name)
            elif name.endswith('self_interaction_2') and not any(n.startswith(name.split('_')[0]) and 'gate' in n for n, _ in self.model.named_children()):
                # Sometimes the last layer might not have a gate if it only outputs scalars
                # We check this later, but gate_modules usually handles standard
                # SevenNet.
                pass

        # More robust: find blocks by index
        num_layers = 0
        for name in self.model._modules.keys():
            if name.endswith('convolution'):
                num_layers += 1

        idx = layer_index
        if idx < 0:
            idx = num_layers + idx

        if idx < 0 or idx >= num_layers:
            raise ValueError(
                f"layer_index {layer_index} is out of bounds for model with {num_layers} layers.")

        # Find the last module for the specific layer block (e.g. starting with
        # '{idx}_')
        target_name = None
        for name in self.model._modules.keys():
            if name.startswith(f"{idx}_"):
                target_name = name

        if target_name is None:
            raise ValueError(f"Could not find module block for layer index {idx}")

        return target_name

    def create(self, atoms: ase.Atoms) -> np.ndarray:
        """
        Extracts L=0 descriptors for each atom in the given Atoms object.

        Args:
            atoms: An ase.Atoms object.

        Returns:
            A numpy array of shape (N_atoms, N_features).
        """
        # Convert to SevenNet graph format
        data_dict = unlabeled_atoms_to_graph(atoms, self.cutoff, with_shift=False)
        data = AtomGraphData.from_numpy_dict(data_dict)
        data.to(self.device)

        # Forward pass up to the target module
        with torch.no_grad():
            data = self.model._preprocess(data)
            for name, module in self.model.named_children():
                data = module(data)
                if name == self._target_module_name:
                    break

        # The node features are stored in data[KEY.NODE_FEATURE]
        # We need to extract only the L=0 components.
        features = data[KEY.NODE_FEATURE]

        # Determine the irreps of the features at this point to slice correctly
        irreps = Irreps(
            self.model.get_irreps_in(
                self._target_module_name,
                'irreps_out'))

        # e3nn stores irreps as concatenated tensors.
        # We want to extract only the slices corresponding to L=0 (0e or 0o)
        l0_indices = []
        current_idx = 0
        for mul, ir in irreps:
            dim = mul * ir.dim
            if ir.l == 0:
                l0_indices.extend(range(current_idx, current_idx + dim))
            current_idx += dim

        features_l0 = features[:, l0_indices].detach().cpu().numpy()

        if self.features_dim is not None and features_l0.shape[1] != self.features_dim:
            warnings.warn(
                f"Extracted feature dimension ({
                    features_l0.shape[1]}) does not match " f"expected features_dim ({
                    self.features_dim}).")

        return features_l0


def main():
    import argparse
    import pandas as pd
    import ase.io

    parser = argparse.ArgumentParser(
        description="Extract SevenNet descriptors from atomic structures.")
    parser.add_argument("--model", type=str, default="7net-0",
                        help="Model name or checkpoint path.")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input structure file (e.g., structures.extxyz).")
    parser.add_argument(
        "--output",
        type=str,
        default="features.csv",
        help="Output CSV file.")
    parser.add_argument(
        "--layer",
        type=int,
        default=-1,
        help="Convolution layer index to extract from (-1 for the last).")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device to use ('cpu', 'cuda', 'auto').")

    args = parser.parse_args()

    print(f"Loading model '{args.model}'...")
    desc = SevenNetDescriptor(
        model=args.model,
        device=args.device,
        layer_index=args.layer)

    print(f"Reading structures from '{args.input}'...")
    atoms_list = ase.io.read(args.input, index=":")
    if not isinstance(atoms_list, list):
        atoms_list = [atoms_list]

    print(
        f"Extracting features from layer {
            args.layer} for {
            len(atoms_list)} structures...")

    all_features = []

    for i, atoms in enumerate(atoms_list):
        feats = desc.create(atoms)
        symbols = atoms.get_chemical_symbols()

        for atom_idx, (symbol, feat) in enumerate(zip(symbols, feats)):
            row = {
                "structure_index": i,
                "atom_index": atom_idx,
                "element": symbol,
            }
            # Add feature columns
            for f_idx, f_val in enumerate(feat):
                row[f"feature_{f_idx}"] = f_val

            all_features.append(row)

    df = pd.DataFrame(all_features)
    df.to_csv(args.output, index=False)
    print(f"Successfully saved features to '{args.output}'. Shape: {df.shape}")


if __name__ == "__main__":
    main()
