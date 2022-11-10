import logging
from logging import info
from pathlib import Path
import mlflow as mf
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig
from transformers import AutoTokenizer, BertTokenizerFast


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


def run_exists(run_id: str) -> bool:
    try:
        mf.get_run(run_id)
    except mf.exceptions.MlflowException:
        return False
    return True


def bootup_pipeline_component(params: DictConfig) -> (str, str):
    logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")

    info(f'Working directory: {Path.cwd()}')
    log_file = HydraConfig.get().job_logging.handlers.file.filename
    info(f'Log file: {log_file}')
    dot_hydra = f'{HydraConfig.get().run.dir}/{HydraConfig.get().output_subdir}'
    info(f'Hydra output sub-directory: {dot_hydra}')

    # Note: to use Databricks, first make a Databricks account, install databricks-cli and run  `databricks configure`
    tracking_uri = 'databricks' if params.main.use_databricks else '../../mlruns'
    if not params.main.use_databricks:  # If it is a local path, MLFlow wants it to be absolute to function correctly
        tracking_uri = str(Path(tracking_uri).absolute())
    info(f'Tracking info will go to: {tracking_uri}')
    mf.set_tracking_uri(tracking_uri)
    experiment_name = params.main.experiment_name
    mf.set_experiment(experiment_name)
    info(f'Experiment name is: {experiment_name}')
    return log_file, dot_hydra


tokenizer_from_pretrained = AutoTokenizer.from_pretrained("bert-base-cased")


def tokenize_dataset(data) -> BertTokenizerFast:  # TODO move it into utils
    # Keys of the returned dictionary will be added to the dataset as columns
    return tokenizer_from_pretrained(data["text"], padding='longest', truncation=True)
