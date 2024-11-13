import random
import os
import sys
import logging
import fnmatch

from typing import List

from dataclasses import dataclass, field
from jobflow import Maker, job, Response, Flow

from utils.al_md import ALMDLogger
from utils.io import MACEInputGenerator, save_to_yaml
from utils.ffile import search_file

from jobs.job import md_cal,train_mace

@dataclass
class MaceMaker(Maker):

    n_configs: int = 5

    def make(self) -> list[str]:

        config_files = []
        for i in range(1, self.n_configs + 1):
            random_seed = random.randint(0, 10000)
            generator = MACEInputGenerator(seed=random_seed)
            filename = f'active_learning/config/config-{i}.yml'
            save_to_yaml(generator, filename)
            config_files.append(filename)

        train_jobs = [train_mace(config_file) for config_file in config_files]

        return train_jobs

@dataclass
class MdMaker(Maker):

    temperature: int = 300
    structure_path: str = "active_learning/MD_data"


    def make(self,model_path:list):
        structure_list = []

        for file in os.listdir(self.structure_path):
            if fnmatch.fnmatch(file,'*.xyz'):
                structure_list.append(file)

        md_jobs = [md_cal(model_path,structure,self.temperature) for structure in structure_list]

        return md_jobs
