from monty.json import MSONable
import yaml
import abc
import random

class InputGenerator(MSONable):
    @abc.abstractmethod
    def get_input_set(self, *args, **kwargs):
        raise NotImplementedError(f"get_input_set has not been implemented in {type(self).__name__}")

class MACEInputGenerator(InputGenerator):
    def __init__(self, seed=None, **kwargs):
        self.config = {
            "name": f"RuO2_{seed}_mace" if seed else "RuO2_default_mace",
            "train_file": "active_learning/MACE_data/train.xyz",
            "E0s": "isolated",
            "energy_key": "dft_energy",
            "forces_key": "dft_forces",
            "stress_key": "dft_stress",
            "model_dir": "active_learning/MACE_models",
            "log_dir": "active_learning/MACE_models",
            "checkpoints_dir": "active_learning/MACE_models",
            "results_dir": "active_learning/MACE_models",
            "forces_weight": 1000,              # int
            "energy_weight": 10,                # int
            "compute_stress": True,             # bool
            "model": "MACE",                    # str
            "num_channels": 128,                 # int
            "max_L": 2,                         # int
            "r_max": 5.0,                       # float
            "batch_size": 2,                   # int
            "correlation": 3,                   # int
            "num_interactions": 2,              # int
            "max_num_epochs": 300,               # int
            "swa": True,                        # bool
            "start_swa": 220,                   # int
            "swa_forces_weight": 40,            # int
            "patience": 15,                     # int
            "ema": True,                        # bool
            "ema_decay": 0.99,                  # float
            "error_table": "PerAtomRMSEstressvirials",  # str
            "default_dtype": "float64",         # str
            "amsgrad": True,                    # bool
            "device": "cuda",                   # str
            "seed": seed if seed is not None else random.randint(0, 10000)
        }

        self.config.update(kwargs)

    def get_input_set(self):
        return self.config

def save_to_yaml(generator: InputGenerator, filepath: str):
    config = generator.get_input_set()
    with open(filepath, "w") as file:
        yaml.dump(config, file, default_flow_style=False)
