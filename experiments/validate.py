import hydra
import jax
import omegaconf

from memento.trainers.validation import validate
from memento.utils.logger import (EnsembleLogger, NeptuneLogger, TerminalLogger,
                                create_logger, create_subdirectory_path)


@hydra.main(
    config_path="config",
    version_base=None,
    config_name="config_exp",
)
def run(cfg: omegaconf.DictConfig) -> None:
    # Check and configure the available devices.
    validation_cfg = cfg.validation

    available_devices = len(jax.local_devices())
    if validation_cfg.num_devices < 0:
        validation_cfg.num_devices = available_devices
        print(f"Using {available_devices} available device(s).")
    else:
        assert (
            available_devices >= validation_cfg.num_devices
        ), f"{validation_cfg.num_devices} devices requested but only {available_devices} available."

    # create a logger
    cfg.logger.neptune.name = "fastrl-" + cfg.logger.neptune.name

    neptune_tags = ("fastrl",) # "final-exp")
    logger = create_logger(cfg, tags=neptune_tags)

    # init the random key
    metrics = validate(
        cfg=validation_cfg,
    )

    metrics = {f"validate/{k}": v for (k, v) in metrics.items()}
    logger.write(metrics)


if __name__ == "__main__":
    run()
