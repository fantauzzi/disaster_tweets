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


@hydra.main(version_base=None, config_path="../../config", config_name="params")
def main(params: DictConfig) -> None:
    log_file, dot_hydra = bootup_pipeline_component(params=params, path_to_mlruns='../../mlruns')
    trained_model = params.main.trained_model

    # Fetch and load into memory the fine-tuned model ready for inference, if available, otherwise the pre-trained
    # model, to be fine-tuned before inference
    if trained_model is None:
        error('No fine-tuned model was provided in parameter main.trainedModel; cannot proceed, exiting.')
        exit(-1)
    info(f'Taking fine-tuned model from {trained_model}')
    local_trained_model = mf.artifacts.download_artifacts(artifact_uri=trained_model)
    if local_trained_model is None or not Path(local_trained_model).exists():
        error("Couldn't obtain the fine-tuned model; cannot proceed, exiting.")
    info(f'Loading fine-tuned model from {local_trained_model}')
    model = TFAutoModelForSequenceClassification.from_pretrained(f'{local_trained_model}')

    test_data = f'../../{params.main.val_data}'
    dataset_test = Dataset.load_from_disk(test_data)

    # tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    tokenizer = tokenizer_from_pretrained
    dataset_test = dataset_test.map(tokenize_dataset)
    batch_size = params.train.batch_size
    tf_dataset_test = model.prepare_tf_dataset(dataset_test, batch_size=batch_size, shuffle=False, tokenizer=tokenizer)

    if params.main.run_id is not None:
        info(f'Trying to resume run with ID {params.main.run_id}')
    with mf.start_run(run_id=params.main.run_id):  # Start the MLFlow run
        # Retrieve the run name MLflow has assigned. It's a tad convoluted, but it is what MLFlow allows
        mlflow_client = mf.MlflowClient()
        mlflow_run_data = mlflow_client.get_run(mf.active_run().info.run_id).data
        info(f'Started MLFlow run: {mlflow_run_data.tags["mlflow.runName"]}')

        unfolded_params = unfold_config(params)
        mf.log_params(unfolded_params)

        """ Hugging Face create_optimizer() has a mandatory parameter num_train_steps, bnut the actual value
        here is not relevant, as we are only testing the fine-tuned model, no model training here """
        num_epochs = 1  # The no. of epochs is not important here, as we only test the model
        batches_per_epoch = len(dataset_test) // batch_size
        total_train_steps = int(batches_per_epoch * num_epochs)

        optimizer, schedule = create_optimizer(init_lr=2e-5, num_warmup_steps=0, num_train_steps=total_train_steps)

        model.compile(optimizer=optimizer, metrics=['acc'])

        # Test the fine-tuned model
        test_batch_size = params.test.batch_size
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

Switch from multiple runs within one experiment to multiple experiments and every MLFlow project gets its own run.

Ensure runs ar reproducible (usage of random number generator seed) 
'''
