import sys
from pathlib import Path
import logging
from logging import warning, info, error
from datasets import Dataset, Value, ClassLabel, Features
import mlflow as mf
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

sys.path.append('..')
from utils.common import bootup_pipeline_component


@hydra.main(version_base=None, config_path="../../config", config_name="params")
def main(params: DictConfig) -> None:
    # Note, Hydra by default sends all log messages both to the console and a log file.
    # See https://hydra.cc/docs/1.2/tutorials/basic/running_your_app/logging/
    # Hydra produces the log file only when initialized via the decorator, not if initialized with the Compose API
    # See https://github.com/facebookresearch/hydra/issues/2456
    log_file, dot_hydra = bootup_pipeline_component(params)

    seed = params.main.seed

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
    if params.main.run_id is not None:
        info(f'Trying to resume run with ID {params.main.run_id}')
    with mf.start_run(run_id=params.main.run_id):  # Start the MLFlow run
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


'''
