import argparse
import os
import gc
import re
import numpy as np
import torch
import ase.io
from ase.stress import full_3x3_to_voigt_6_stress
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm
from pathlib import Path

from sevenn.calculator import SevenNetCalculator

def evaluate_structures(atoms_list, calc):
    """Extracts DFT targets and MLP predictions into structured dictionaries."""
    dft = {'e': [], 'f': [], 's': [], 'bec_diag': [], 'bec_off': [], 'bec_full': [], 'dielectric_diag': [], 'dielectric_off': [], 'dielectric_full': [], 'symbols': []}
    mlp = {'e': [], 'f': [], 's': [], 'bec_diag': [], 'bec_off': [], 'bec_full': [], 'dielectric_diag': [], 'dielectric_off': [], 'dielectric_full': []}

    species_data = {}
    diag_mask = np.eye(3, dtype=bool)

    for atoms in tqdm(atoms_list, desc="Running Inference"):
        n = len(atoms)
        symbols = atoms.get_chemical_symbols()
        dft['symbols'].extend(symbols)

        # --- 1. Extraction DFT (Ground Truth) ---
        e_dft = (atoms.get_potential_energy() / n) if atoms.calc else (atoms.info.get('energy', np.nan) / n)
        f_dft = atoms.get_forces() if atoms.calc else atoms.arrays.get('forces', np.full((n, 3), np.nan))

        # Robust Stress/Virial Extraction
        if atoms.calc:
            s_dft = atoms.get_stress()
        else:
            if 'stress' in atoms.info:
                s_raw = np.array(atoms.info['stress'])
            elif 'virial' in atoms.info:
                s_raw = np.array(atoms.info['virial']) / atoms.get_volume()
            else:
                s_raw = np.full(6, np.nan)

            if s_raw.size == 9:
                s_dft = full_3x3_to_voigt_6_stress(s_raw.reshape(3, 3))
            elif s_raw.size == 6:
                s_dft = s_raw.flatten()
            else:
                s_dft = np.full(6, np.nan)

        bec_dft = atoms.calc.results.get('born_effective_charges') if atoms.calc else atoms.arrays.get('born_effective_charges')
        bec_dft = np.zeros((n, 3, 3)) if bec_dft is None else bec_dft.reshape(-1, 3, 3)

        dielectric_dft = atoms.calc.results.get('dielectric_tensor') if atoms.calc else atoms.info.get('dielectric_tensor')
        if dielectric_dft is not None:
            dielectric_dft = np.array(dielectric_dft).reshape(3, 3)
            dft['dielectric_full'].append(dielectric_dft)
            dft['dielectric_diag'].append(dielectric_dft[diag_mask])
            dft['dielectric_off'].append(dielectric_dft[~diag_mask])
        else:
            dft['dielectric_full'].append(np.full((3, 3), np.nan))
            dft['dielectric_diag'].append(np.full(3, np.nan))
            dft['dielectric_off'].append(np.full(6, np.nan))

        dft['e'].append(e_dft)
        dft['f'].append(f_dft)
        dft['s'].append(s_dft)
        dft['bec_full'].append(bec_dft)
        dft['bec_diag'].append(bec_dft[:, diag_mask])
        dft['bec_off'].append(bec_dft[:, ~diag_mask])

        # --- 2. Extraction MLP (Prediction) ---
        clean_atoms = atoms.copy()
        clean_atoms.calc = calc

        e_mlp = clean_atoms.get_potential_energy() / n
        f_mlp = clean_atoms.get_forces()

        try:
            s_mlp = clean_atoms.get_stress()
        except:
            s_mlp = np.full(6, np.nan)

        bec_mlp = clean_atoms.calc.results.get('born_effective_charges',
                  clean_atoms.calc.results.get('BornEffectiveCharges', np.zeros((n, 3, 3)))).copy()

        dielectric_mlp = clean_atoms.calc.results.get('dielectric_tensor')
        if dielectric_mlp is not None:
            dielectric_mlp = np.array(dielectric_mlp).reshape(3, 3)
            mlp['dielectric_full'].append(dielectric_mlp)
            mlp['dielectric_diag'].append(dielectric_mlp[diag_mask])
            mlp['dielectric_off'].append(dielectric_mlp[~diag_mask])
        else:
            mlp['dielectric_full'].append(np.full((3, 3), np.nan))
            mlp['dielectric_diag'].append(np.full(3, np.nan))
            mlp['dielectric_off'].append(np.full(6, np.nan))

        mlp['e'].append(e_mlp)
        mlp['f'].append(f_mlp)
        mlp['s'].append(s_mlp)
        mlp['bec_diag'].append(bec_mlp[:, diag_mask])
        mlp['bec_off'].append(bec_mlp[:, ~diag_mask])
        mlp['bec_full'].append(bec_mlp)

        clean_atoms.calc = None

        # --- 3. Species-Resolved BEC Errors & Targets ---
        for i, sym in enumerate(symbols):
            if sym not in species_data:
                species_data[sym] = {'errors': [], 'dft_diag': []}

            error_tensor = bec_mlp[i] - bec_dft[i]
            species_data[sym]['errors'].append(error_tensor.flatten())
            species_data[sym]['dft_diag'].append(bec_dft[i][diag_mask])

    # Final Concatenation
    dft['symbols'] = np.array(dft['symbols'])
    for k in dft:
        if k == 'symbols': continue
        if k in ['e', 's', 'dielectric_full', 'dielectric_diag', 'dielectric_off']:
            dft[k] = np.array(dft[k])
            mlp[k] = np.array(mlp[k])
        else:
            dft[k] = np.concatenate(dft[k], axis=0) if len(dft[k]) > 0 else np.array([])
            mlp[k] = np.concatenate(mlp[k], axis=0) if len(mlp[k]) > 0 else np.array([])

    return dft, mlp, species_data

def print_summary_metrics(dft, mlp):
    """Prints a clean summary of both metric styles directly to the terminal."""
    print("\n" + "="*75)
    print("                      GLOBAL INFERENCE METRICS")
    print("="*75)

    # Energy
    valid_e = ~np.isnan(dft['e']) & ~np.isnan(mlp['e'])
    if np.any(valid_e):
        rmse_e = np.sqrt(np.mean((dft['e'][valid_e] - mlp['e'][valid_e])**2))
        mae_e = np.mean(np.abs(dft['e'][valid_e] - mlp['e'][valid_e]))
        print(f"Energy (eV/atom)     | RMSE: {rmse_e:.4f}  | MAE: {mae_e:.4f}")

    # Forces
    if 'f' in dft and len(dft['f']) > 0:
        valid_f = ~np.isnan(dft['f']).any(axis=1) & ~np.isnan(mlp['f']).any(axis=1)
        if np.any(valid_f):
            f_d, f_m = dft['f'][valid_f].flatten(), mlp['f'][valid_f].flatten()
            rmse_f_acad = np.sqrt(np.mean((f_d - f_m)**2))
            rmse_f_7n = np.sqrt(np.mean((f_d - f_m)**2) * 3)
            print(f"Forces (eV/A)        | RMSE (Academic): {rmse_f_acad:.4f}  | RMSE (SevenNet): {rmse_f_7n:.4f}")

    # Stress
    if 's' in dft and len(dft['s']) > 0:
        valid_s = ~np.isnan(dft['s']).any(axis=1) & ~np.isnan(mlp['s']).any(axis=1)
        if np.any(valid_s):
            EV_A3_TO_GPA = 160.21766208
            s_d = dft['s'][valid_s] * EV_A3_TO_GPA
            s_m = mlp['s'][valid_s] * EV_A3_TO_GPA

            rmse_s_7n = np.sqrt(np.mean((s_d.flatten() - s_m.flatten())**2) * 6)

            diff = s_d - s_m
            sq_err = diff**2
            sq_err[:, 3:] *= 2
            rmse_s_acad = np.sqrt(np.sum(sq_err) / (9 * len(s_d)))

            print(f"Stress (GPa)         | RMSE (Academic): {rmse_s_acad:.4f}  | RMSE (SevenNet): {rmse_s_7n:.4f}")

    # BEC Full Tensor
    if 'bec_full' in dft and len(dft['bec_full']) > 0:
        valid_bec = ~np.isnan(dft['bec_full']).any(axis=(1,2)) & ~np.isnan(mlp['bec_full']).any(axis=(1,2))
        if np.any(valid_bec):
            b_d, b_m = dft['bec_full'][valid_bec].flatten(), mlp['bec_full'][valid_bec].flatten()
            rmse_b = np.sqrt(np.mean((b_d - b_m)**2))
            print(f"Born Charges (e)     | RMSE (Component): {rmse_b:.4f}")

    # Dielectric Tensor
    if 'dielectric_full' in dft and len(dft['dielectric_full']) > 0:
        valid_diel = ~np.isnan(dft['dielectric_full']).any(axis=(1,2)) & ~np.isnan(mlp['dielectric_full']).any(axis=(1,2))
        if np.any(valid_diel):
            d_d, d_m = dft['dielectric_full'][valid_diel].flatten(), mlp['dielectric_full'][valid_diel].flatten()
            rmse_d = np.sqrt(np.mean((d_d - d_m)**2))
            print(f"Dielectric Tensor    | RMSE (Component): {rmse_d:.4f}")

    print("="*75 + "\n")


def save_inference_plot(dft, mlp, species_data, output_filename, model_name, data_name, style="academic"):
    print(f'Generating {style.capitalize()} global parity plot...')

    fig = plt.figure(figsize=(12, 8), dpi=300)
    title_str = f"SevenNet Inference Dashboard ({style.capitalize()} Metrics)\nModel: {model_name}  |  Dataset: {data_name}"
    fig.suptitle(title_str, fontsize=16, fontweight='bold')

    gs_main = gridspec.GridSpec(2, 3, figure=fig)

    axes = [
        fig.add_subplot(gs_main[0, 0]),
        fig.add_subplot(gs_main[0, 1]),
        fig.add_subplot(gs_main[0, 2]),
        fig.add_subplot(gs_main[1, 0]),
        fig.add_subplot(gs_main[1, 1]),
        fig.add_subplot(gs_main[1, 2])
    ]

    def add_metrics_and_line(ax, x, y, metric_context='default'):
        if x.ndim == 2:
            valid = ~np.isnan(x).any(axis=1) & ~np.isnan(y).any(axis=1)
        else:
            valid = ~np.isnan(x) & ~np.isnan(y)

        if not np.any(valid): return

        x_val, y_val = x[valid], y[valid]
        x_flat, y_flat = x_val.flatten(), y_val.flatten()

        lims = [np.min([x_flat, y_flat]), np.max([x_flat, y_flat])]
        ax.plot(lims, lims, 'grey', ls='--', lw=1, zorder=0)

        # Determine Metric Logic based on requested Style
        n_comp = 1
        is_tensor_stress = False

        if style == 'sevennet':
            if metric_context == 'force': n_comp = 3
            elif metric_context == 'stress': n_comp = 6
        elif style == 'academic':
            if metric_context == 'stress': is_tensor_stress = True

        # Compute Math
        if is_tensor_stress:
            diff = x_val - y_val
            sq_err = diff**2
            sq_err[:, 3:] *= 2
            rmse = np.sqrt(np.sum(sq_err) / (9 * len(x_val)))

            abs_err = np.abs(diff)
            abs_err[:, 3:] *= 2
            mae = np.sum(abs_err) / (9 * len(x_val))
        else:
            rmse = np.sqrt(np.mean((x_flat - y_flat)**2) * n_comp)
            mae = np.mean(np.abs(x_flat - y_flat))

        ss_res = np.sum((x_flat - y_flat)**2)
        ss_tot = np.sum((x_flat - np.mean(x_flat))**2)
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-12 else float('nan')

        data_range = np.max(x_flat) - np.min(x_flat)
        if data_range > 1e-12:
            nrmse_pct = (rmse / data_range) * 100
            pct_str = f'\nNRMSE: {nrmse_pct:.2f}%'
        else:
            pct_str = '\nNRMSE: N/A'

        metric_text = f'RMSE: {rmse:.4f}\nMAE:  {mae:.4f}\nR²:      {r2:.6f}{pct_str}'

        ax.text(0.05, 0.95, metric_text, transform=ax.transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
                fontsize=9, zorder=5)

    # --- 1. Energy/atom ---
    axes[0].scatter(dft['e'], mlp['e'], s=30, alpha=0.8, color='tab:blue', edgecolors='black', linewidth=0.8, zorder=2, label='Eval Data')
    axes[0].set_title('Energy per atom')
    axes[0].set_xlabel('DFT (eV/atom)'); axes[0].set_ylabel('SevenNet (eV/atom)')
    axes[0].legend(fontsize=9, loc='lower right')
    add_metrics_and_line(axes[0], dft['e'], mlp['e'], metric_context='energy')
    axes[0].margins(0.15)

    # --- 2. Forces ---
    f_dft_flat = dft['f'].flatten() if dft['f'].ndim == 2 else dft['f']
    f_mlp_flat = mlp['f'].flatten() if mlp['f'].ndim == 2 else mlp['f']
    axes[1].scatter(f_dft_flat, f_mlp_flat, s=5, alpha=0.8, color='tab:green', zorder=2)
    axes[1].set_title('Forces (all components)')
    axes[1].set_xlabel('DFT (eV/Å)'); axes[1].set_ylabel('SevenNet (eV/Å)')
    add_metrics_and_line(axes[1], dft['f'], mlp['f'], metric_context='force')

    # --- 3. Stress (in GPa) ---
    EV_A3_TO_GPA = 160.217662
    s_dft = dft['s'] * EV_A3_TO_GPA
    s_mlp = mlp['s'] * EV_A3_TO_GPA
    valid_s = ~np.isnan(s_dft).any(axis=1) & ~np.isnan(s_mlp).any(axis=1)
    if np.any(valid_s):
        axes[2].scatter(s_dft[:, :3][valid_s].flatten(), s_mlp[:, :3][valid_s].flatten(), s=15, alpha=0.6, color='tab:red', label='Diag (xx,yy,zz)', zorder=2)
        axes[2].scatter(s_dft[:, 3:][valid_s].flatten(), s_mlp[:, 3:][valid_s].flatten(), s=15, alpha=0.6, color='tab:orange', marker='^', label='Off-Diag', zorder=2)
        axes[2].legend(fontsize=9, loc='lower right')

    axes[2].set_title('Stress Components')
    axes[2].set_xlabel('DFT (GPa)'); axes[2].set_ylabel('SevenNet (GPa)')
    add_metrics_and_line(axes[2], s_dft, s_mlp, metric_context='stress')

    # --- 4. BEC Diagonal ---
    colors_diag = ['tab:red', 'tab:green', 'tab:blue']
    labels_diag = ['xx', 'yy', 'zz']
    for i in range(3):
        axes[3].scatter(dft['bec_diag'][:, i], mlp['bec_diag'][:, i], s=5, alpha=0.5, color=colors_diag[i], label=labels_diag[i], zorder=2)
    axes[3].set_title('BEC Diagonal')
    axes[3].set_xlabel('DFPT (e)'); axes[3].set_ylabel('SevenNet (e)')
    axes[3].legend(fontsize=9)
    add_metrics_and_line(axes[3], dft['bec_diag'], mlp['bec_diag'], metric_context='bec')

    # --- 5. BEC Off-Diagonal ---
    colors_off = ['tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
    labels_off = ['xy', 'xz', 'yx', 'yz', 'zx', 'zy']
    for i in range(6):
        axes[4].scatter(dft['bec_off'][:, i], mlp['bec_off'][:, i], s=5, alpha=0.5, color=colors_off[i], label=labels_off[i], zorder=2)
    axes[4].set_title('BEC Off-Diagonal')
    axes[4].set_xlabel('DFPT (e)'); axes[4].set_ylabel('SevenNet (e)')
    axes[4].legend(fontsize=8, ncol=2)
    add_metrics_and_line(axes[4], dft['bec_off'], mlp['bec_off'], metric_context='bec')

    # --- 6. Dielectric Tensor ---
    valid_diel = ~np.isnan(dft['dielectric_full']).any(axis=(1,2)) & ~np.isnan(mlp['dielectric_full']).any(axis=(1,2))
    if np.any(valid_diel):
        axes[5].scatter(dft['dielectric_diag'][valid_diel].flatten(), mlp['dielectric_diag'][valid_diel].flatten(), s=15, alpha=0.6, color='tab:red', label='Diag (xx,yy,zz)', zorder=2)
        axes[5].scatter(dft['dielectric_off'][valid_diel].flatten(), mlp['dielectric_off'][valid_diel].flatten(), s=15, alpha=0.6, color='tab:orange', marker='^', label='Off-Diag', zorder=2)
        axes[5].legend(fontsize=9, loc='lower right')

    axes[5].set_title('Dielectric Tensor Components')
    axes[5].set_xlabel('DFPT'); axes[5].set_ylabel('SevenNet')
    add_metrics_and_line(axes[5], dft['dielectric_full'][valid_diel].flatten(), mlp['dielectric_full'][valid_diel].flatten(), metric_context='dielectric')

    plt.tight_layout()
    plt.savefig(output_filename, bbox_inches='tight')
    plt.close()
    print(f"Saved {style} dashboard to {output_filename}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate a SevenNet checkpoint and generate your custom parity dashboard.")
    parser.add_argument("model_path", type=str, help="Path to the trained checkpoint (.pth)")
    parser.add_argument("xyz_path", type=str, help="Path to the dataset (.xyz) to evaluate")
    parser.add_argument("--out", type=str, default="inference_dashboard.png", help="Base output filename for the plots")
    args = parser.parse_args()

    data_name = os.path.splitext(os.path.basename(args.xyz_path))[0]
    abs_model_path = os.path.abspath(args.model_path)
    parent_dir = os.path.basename(os.path.dirname(abs_model_path))
    grandparent_dir = os.path.basename(os.path.dirname(os.path.dirname(abs_model_path)))

    raw_model_name = grandparent_dir if "Run_" in parent_dir else parent_dir
    model_name = re.sub(r'^\d{8}_\d{3}_', '', raw_model_name)
    if not model_name: model_name = raw_model_name

    print(f"{'='*50}")
    print(f"Model File: {args.model_path}")
    print(f"Model Name: {model_name}")
    print(f"Data File:  {args.xyz_path}")
    print(f"Data Name:  {data_name}")
    print(f"{'='*50}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    calc = SevenNetCalculator(model=args.model_path, device=device, enable_flash=True)

    atoms_list = ase.io.read(args.xyz_path, index=':')
    dft_data, mlp_data, species_data = evaluate_structures(atoms_list, calc)

    # Print the terminal summary
    print_summary_metrics(dft_data, mlp_data)

    # Setup filenames for both styles
    base_out = args.out
    if base_out.endswith(".png"):
        out_acad = base_out.replace(".png", "_Academic.png")
        out_7n = base_out.replace(".png", "_SevenNet.png")
    else:
        out_acad = base_out + "_Academic.png"
        out_7n = base_out + "_SevenNet.png"

    # Generate the two plots
    save_inference_plot(dft_data, mlp_data, species_data, out_acad, model_name, data_name, style="academic")
    save_inference_plot(dft_data, mlp_data, species_data, out_7n, model_name, data_name, style="sevennet")

    del calc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
