import sys
from pathlib import Path
from logging import warning, info, error
from datasets import Dataset
from transformers import TFAutoModelForSequenceClassification
from transformers import create_optimizer
import mlflow as mf
import hydra
from omegaconf import DictConfig

sys.path.append("..")
from utils.common import unfold_config, bootup_pipeline_component, tokenize_dataset, tokenizer_from_pretrained

# cd ../..; mlflow run --experiment-name  /disaster-tweets src/train/

@hydra.main(version_base=None, config_path="../../config", config_name="params")
def main(params: DictConfig) -> None:
    log_file, dot_hydra = bootup_pipeline_component(params)
    trained_model = params.main.trained_model

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
    val_data = f'../../{params.main.test_data}'
    dataset = Dataset.load_from_disk(train_data)
    dataset_val = Dataset.load_from_disk(val_data)

    # Tokenize the tweets and prepare Tensorflow datasets
    """def tokenize_dataset(data):
        # Keys of the returned dictionary will be added to the dataset as columns
        return tokenizer(data["text"], padding='longest', truncation=True)"""

    # tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    tokenizer = tokenizer_from_pretrained  # TODO tokenizer is redundant
    dataset = dataset.map(tokenize_dataset)
    dataset_val = dataset_val.map(tokenize_dataset)
    batch_size = params.train.batch_size
    tf_dataset = model.prepare_tf_dataset(dataset, batch_size=batch_size, shuffle=True, tokenizer=tokenizer)
    tf_dataset_val = model.prepare_tf_dataset(dataset_val, batch_size=batch_size, shuffle=False, tokenizer=tokenizer)

    with mf.start_run():  # Start the MLFlow run
        # Retrieve the run name MLflow has assigned. It's a tad convoluted, but it is what MLFlow allows
        mlflow_client = mf.MlflowClient()
        mlflow_run_data = mlflow_client.get_run(mf.active_run().info.run_id).data
        info(f'Started MLFlow run: {mlflow_run_data.tags["mlflow.runName"]}')

        # mf.log_artifact(local_path=f'{get_original_cwd()}/params.yaml')
        unfolded_params = unfold_config(params)
        mf.log_params(unfolded_params)
        # mf.tensorflow.autolog(log_models=False)

        num_epochs = params.train.num_epochs
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
        else:
            info('Fine-tuned model is already available, no model needs to be fine-tuned here.')

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
