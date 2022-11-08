from pathlib import Path
import logging
from logging import warning, info, error
from datasets import Dataset, Value, ClassLabel, Features
import mlflow as mf
import hydra
from hydra.utils import get_original_cwd
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf


def unfold_config(config: DictConfig | dict) -> dict[str, str]:
    """
    Takes a DictConfig, or a dict obtained from a DictConfig, and converts it to a dict with one key-value pair
    for every parameter, where the grouping of keys from the DictConfig is replaced by concatenating all the keys
    with a dot.
    :param config: the given DictConfig, or the given DictConfig cast to a dict.
    :return: a dictionary with the result of the translation.
    """

    def unfold_config_as_list(config: DictConfig | dict) -> list[str]:
        res = []
        for key, value in config.items():
            if isinstance(value, dict) or isinstance(value, DictConfig):
                embedded_res = unfold_config_as_list(value)
                res.extend([f'{key}.{item}' for item in embedded_res])
            else:
                res.append(f'{key} {value}')
        return res

    res = unfold_config_as_list(config)
    res = {item[:item.rfind(' ')]: item[item.rfind(' ') + 1:] for item in res}
    res = dict(sorted(res.items()))
    return res


@hydra.main(version_base=None, config_path="../../config", config_name="params")
def main(params: DictConfig) -> None:
    # Note, Hydra by default sends all log messages both to the console and a log file.
    # See https://hydra.cc/docs/1.2/tutorials/basic/running_your_app/logging/
    logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")

    info(f'Working directory: {Path.cwd()}')
    log_file = HydraConfig.get().job_logging.handlers.file.filename
    info(f'Log file: {log_file}')
    dot_hydra = f'{HydraConfig.get().run.dir}/{HydraConfig.get().output_subdir}'
    info(f'Hydra output sub-directory: {dot_hydra}')

    seed = params.main.seed
    # Note: to use Databricks, first make a Databricks account, install databricks-cli and run  `databricks configure`

    tracking_uri = 'databricks' if params.main.use_databricks else '../../mlruns'
    if not params.main.use_databricks:  # If it is a local path, MLFlow wants it to be absolute to function correctly
        tracking_uri = str(Path(tracking_uri).absolute())
    info(f'Tracking info will go to: {tracking_uri}')
    mf.set_tracking_uri(tracking_uri)
    experiment_name = params.main.experiment_name
    mf.set_experiment(experiment_name)
    info(f'Experiment name is: {experiment_name}')

    # Load and split the dataset
    info('Loading dataset ...')
    features = Features({'id': Value(dtype='string', id=None),
                         'keyword': Value(dtype='string', id=None),
                         'location': Value(dtype='string', id=None),
                         'text': Value(dtype='string', id=None),
                         'target': ClassLabel(num_classes=2, names=['0', '1'], id=None)})

    dataset = Dataset.from_csv(f'../../{params.main.source_data}', features=features)
    dataset = dataset.remove_columns(['id', 'keyword', 'location'])
    dataset = dataset.rename_column('target', 'label')
    dataset = dataset.train_test_split(test_size=.2, stratify_by_column='label', seed=seed)
    dataset_test = dataset['test']
    dataset = dataset['train']
    dataset = dataset.train_test_split(test_size=.2, stratify_by_column='label', seed=seed)
    dataset_val = dataset['test']
    dataset = dataset['train']
    info(f'Train set size: {len(dataset)}   Val. set size: {len(dataset_val)}   Test set size: {len(dataset_test)}')

    train_data = f'../../{params.main.train_data}'
    test_data = f'../../{params.main.val_data}'
    val_data = f'../../{params.main.test_data}'
    dataset.save_to_disk(train_data)
    dataset_test.save_to_disk(test_data)
    dataset_val.save_to_disk(val_data)
    with mf.start_run():  # Start the MLFlow run
        # Retrieve the run name MLflow has assigned. It's a tad convoluted, but it is what MLFlow allows
        mlflow_client = mf.MlflowClient()
        mlflow_run_data = mlflow_client.get_run(mf.active_run().info.run_id).data
        info(f'Started MLFlow run: {mlflow_run_data.tags["mlflow.runName"]}')
        mf.log_artifact(train_data)
        mf.log_artifact(val_data)
        mf.log_artifact(test_data)
        mf.log_artifact(log_file)
        mf.log_artifact(dot_hydra)

if __name__ == '__main__':
    main()

'''
Open Issues
===========
Pass the run name as an optional parameter to the script; if the parameter is present, then use that for the run,
don't start a new run

'''