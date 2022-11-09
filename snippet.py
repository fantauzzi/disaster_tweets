import logging
from logging import info
from pathlib import Path
from hydra import compose, initialize


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")

    initialize(version_base=None, config_path=".")
    params = compose(config_name="params", return_hydra_config=True)
    params.hydra.runtime.output_dir = str(Path.cwd() / 'outputs')  # <== setting params.hydra.runtime.output_dir here
    info(f'Howdy!')
    print(params.hydra.runtime.output_dir)


if __name__ == '__main__':
    main()

    # hydra_config = params.hydra
    # params.hydra.runtime.output_dir = str(Path.cwd() / 'outputs')
    # log_file = hydra_config.job_logging.handlers.file.filename
