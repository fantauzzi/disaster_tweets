from pathlib import Path
import hydra
from omegaconf import DictConfig
import mlflow as mf
from hydra.core.hydra_config import HydraConfig

from utils.common import bootup_pipeline_component


@hydra.main(version_base=None, config_path='../config', config_name='params')
def main(params: DictConfig):
    bootup_pipeline_component(params=params, path_to_mlruns='../mlruns')
    available_steps = ('preprocess', 'train', 'test')
    required_steps = available_steps if params.main.steps in ('all', None) else params.main.steps.split(',')
    for step in required_steps:
        assert step in available_steps
    experiment_name = params.main.experiment_name
    task = HydraConfig.get().overrides.task
    # params_override = hydra_overrides_task_to_dict(task)
    params_override = ' '.join(task)

    if 'preprocess' in required_steps:
        mf.run(uri=str(Path(hydra.utils.get_original_cwd()) / '.'),
               entry_point='preprocess',
               experiment_name=experiment_name,
               parameters={'overrides': params_override})
    if 'train' in required_steps:
        mf.run(uri=str(Path(hydra.utils.get_original_cwd()) / '.'),
               entry_point='train',
               experiment_name=experiment_name,
               parameters={'overrides': params_override})
    if 'test' in required_steps:
        mf.run(uri=str(Path(hydra.utils.get_original_cwd()) / '.'),
               entry_point='test',
               experiment_name=experiment_name,
               parameters={'overrides': params_override})


if __name__ == '__main__':
    main()
