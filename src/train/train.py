from pathlib import Path
import logging
from logging import warning, info, error
from datasets import Dataset, Value, ClassLabel, Features
from transformers import AutoTokenizer, BertModel
from transformers import TFAutoModelForSequenceClassification
from transformers import create_optimizer
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

    trained_model = params['main']['trained_model']
    seed = params['main']['seed']
    # Note: to use Databricks, first make a Databricks account, install databricks-cli and run  `databricks configure`
    tracking_uri = 'databricks' if params['main']['use_databricks'] else '../../mlruns'
    info(f'Tracking info will go to: {tracking_uri}')
    mf.set_tracking_uri(tracking_uri)
    experiment_name = params['main']['experiment_name']
    mf.set_experiment(experiment_name)
    info(f'Experiment name is: {experiment_name}')

    # Load and split the dataset
    info('Loading dataset ...')

    # Fetch and load into memory the fine-tuned model ready for inference, if available, otherwise the pre-trained
    # model, to be fine-tuned before inference
    local_trained_model = None
    if trained_model is not None:  # Fetch the fine-tuned model from MLFlow, if available
        info(f'Taking fine-tuned model from {trained_model}')
        local_trained_model = mf.artifacts.download_artifacts(artifact_uri=trained_model)
    if local_trained_model is not None and Path(local_trained_model).exists():
        info(f'Loading fine-tuned model from {local_trained_model}')
        model = TFAutoModelForSequenceClassification.from_pretrained(f'{local_trained_model}')
    else:  # If no fine-tuned model is available, then fetch the pre-trained (to be fine-tuned) model from Hugging Face
        info('Fetching pre-trained bert-base-cased for fine-tuning')
        model = TFAutoModelForSequenceClassification.from_pretrained("bert-base-cased")

    train_data = f'../../{params.main.train_data}'
    test_data = f'../../{params.main.val_data}'
    val_data = f'../../{params.main.test_data}'
    dataset = Dataset.load_from_disk(train_data)
    dataset_test = Dataset.load_from_disk(test_data)
    dataset_val = Dataset.load_from_disk(val_data)

    # Tokenize the tweets and prepare Tensorflow datasets
    def tokenize_dataset(data):
        # Keys of the returned dictionary will be added to the dataset as columns
        return tokenizer(data["text"], padding='longest', truncation=True)

    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    dataset = dataset.map(tokenize_dataset)
    dataset_val = dataset_val.map(tokenize_dataset)
    dataset_test = dataset_test.map(tokenize_dataset)
    batch_size = params['train']['batch_size']
    tf_dataset = model.prepare_tf_dataset(dataset, batch_size=batch_size, shuffle=True, tokenizer=tokenizer)
    tf_dataset_val = model.prepare_tf_dataset(dataset_val, batch_size=batch_size, shuffle=False, tokenizer=tokenizer)
    tf_dataset_test = model.prepare_tf_dataset(dataset_test, batch_size=batch_size, shuffle=False, tokenizer=tokenizer)

    with mf.start_run():  # Start the MLFlow run
        # Retrieve the run name MLflow has assigned. It's a tad convoluted, but it is what MLFlow allows
        mlflow_client = mf.MlflowClient()
        mlflow_run_data = mlflow_client.get_run(mf.active_run().info.run_id).data
        info(f'Started MLFlow run: {mlflow_run_data.tags["mlflow.runName"]}')

        # mf.log_artifact(local_path=f'{get_original_cwd()}/params.yaml')
        unfolded_params = unfold_config(params)
        mf.log_params(unfolded_params)
        # mf.tensorflow.autolog(log_models=False)

        num_epochs = params['train']['num_epochs']
        batches_per_epoch = len(dataset) // batch_size
        total_train_steps = int(batches_per_epoch * num_epochs)

        optimizer, schedule = create_optimizer(init_lr=2e-5, num_warmup_steps=0, num_train_steps=total_train_steps)

        model.compile(optimizer=optimizer, metrics=['acc'])
        # If the pre-trained model hasn't been fine-tuned, then fine-tune and evaluate it now, and send it to MLFlow
        # as an artifact (not using MLFlow models here)
        if local_trained_model is None or not Path(local_trained_model).exists():
            model_dir = '../../model'
            info('No fine-tuned mode available, therefore fine-tuning the pre-trained model')
            model.fit(tf_dataset, validation_data=tf_dataset_val, epochs=num_epochs)
            model.save_pretrained(save_directory=model_dir)
            info(f'Saved fine-tuned model in directory {model_dir}')
            mf.log_artifact(local_path=model_dir)
            info(
                f'The fine-tuned model has been logged with MLFlow; directory {model_dir} can now be removed, if desired')

        # Test the fine-tuned model
        test_batch_size = params['test']['batch_size']
        info('Testing the fine-tuned model')
        test_res = model.evaluate(tf_dataset_test, batch_size=test_batch_size, return_dict=True)
        info(f'Test results: {test_res}')
        mf.log_metrics(test_res)
        mf.log_artifact(log_file)
        mf.log_artifact(dot_hydra)
        info(f'All relevant artifacts have been successfully logged with MLFlow')


if __name__ == '__main__':
    main()

'''
Open Issues
===========

When fetching the model (as an MLFlow artifact) from Databricks, the model is saved in /tmp, and discarded after the 
run. Multiple runs, even if they all use the same model from Databricks, will re-download the same model over and again.
It should be possible to save the model to a different location than /tmp AND ensure that future runs that reference
the same model use the model already on the local file system, instead of downloading it again. 
'''
