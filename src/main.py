from pathlib import Path
import hydra
from omegaconf import DictConfig
import mlflow as mf


@hydra.main(version_base=None, config_path='config', config_name='params')
def main(config: DictConfig):
    available_steps = ('preprocess', 'train', 'test')
    required_steps = available_steps if config.main.steps in ('all', None) else config.main.steps.split(',')
    for step in required_steps:
        assert step in available_steps
