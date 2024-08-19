import argparse
import os
import subprocess

from torch import __version__ as torch_version

from sevenn import __version__ as sevenn_version

# python wrapper of patch_lammps.sh script
# importlib.resources is correct way to do these things
# but it changes so frequently to use
pair_e3gnn_dir = os.path.abspath(f'{os.path.dirname(__file__)}/../pair_e3gnn')

description = (
    f'sevenn version={sevenn_version}, patch LAMMPS for pair_e3gnn styles'
)


def main(args=None):
    args = cmd_parse_main(args)
    lammps_dir = os.path.abspath(args.lammps_dir)

    print('Patching LAMMPS with the following settings:')
    print('  - LAMMPS source directory:', lammps_dir)

    cxx_standard = '17' if torch_version.startswith('2') else '14'
    if cxx_standard == '17':
        print('  - Torch version >= 2.0 detected, use CXX STANDARD 17')
    else:
        print('  - Torch version < 2.0 detected, use CXX STANDARD 14')

    if args.d3:
        d3_support = '1'
        print('  - D3 support enabled')
    else:
        d3_support = '0'
        print('  - D3 support disabled')

    script = f'{pair_e3gnn_dir}/patch_lammps.sh'
    cmd = f'{script} {lammps_dir} {cxx_standard} {d3_support}'
    res = subprocess.run(cmd.split())
    return res.returncode  # is it meaningless?


def cmd_parse_main(args=None):
    ag = argparse.ArgumentParser(description=description)
    ag.add_argument('lammps_dir', help='Path to LAMMPS source', type=str)
    ag.add_argument('--d3', help='Enable D3 support', action='store_true')
    # cxx_standard is detected automatically
    args = ag.parse_args()
    return args


if __name__ == '__main__':
    main()
