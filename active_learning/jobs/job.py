import random
import os
import sys
import logging
import fnmatch

from typing import List

from ase.io import read
from ase.md.npt import NPT
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
from ase.units import fs
import numpy as np

from mace.cli.run_train import main as mace_run_train_main
from mace.calculators import MACECalculator


from utils.al_md import ALMDLogger
from utils.io import MACEInputGenerator, save_to_yaml
from utils.ffile import search_file

from jobflow import Maker, job, Response, Flow


@job
def train_mace(config_file_path: str) -> str:
    """Training routine for MACE using a configuration file."""
    logging.getLogger().handlers.clear()
    sys.argv = ["program", "--config", config_file_path]
    mace_run_train_main()


@job
def search_file(mace_path):
    mace_model = []
    for file in os.listdir(mace_path):
        if fnmatch.fnmatch(file,'*_mace_swa.model'):
            mace_model.append(os.path.join(mace_path,file))

    return mace_model

@job
def md_cal(model_path,structure,temperature):
    os.makedirs("active_learning/MD_info", exist_ok=True)
    calculator = MACECalculator(model_paths=model_path, device='cuda', default_dtype="float64")

    atoms_in = read(f"active_learning/MD_data/{structure}")
    atoms_in.cell = np.triu(atoms_in.cell)

    output_filename = f"active_learning/MD_info/{structure.split('.')[0]}_NPT_{temperature}K"
    log_filename = output_filename + ".log"
    traj_filename = output_filename + ".traj"

    atoms = atoms_in.copy()
    atoms.calc = calculator
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature, force_temp=True)
    Stationary(atoms)
    ZeroRotation(atoms)

    dyn = NPT(
        atoms,
        1.0 * fs,
        temperature_K=temperature,
        trajectory=traj_filename,
        loginterval=10
        )
    dyn.attach(ALMDLogger(dyn, atoms, log_filename, header=True, active=True, mode="a"), interval=10)
    dyn.run(2000)
