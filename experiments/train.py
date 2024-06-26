import hydra
import jax
import omegaconf

from memento.trainers.trainer import Trainer
from memento.utils.logger import EnsembleLogger, TerminalLogger


def create_logger(cfg) -> EnsembleLogger:
    loggers = []

    if "terminal" in cfg.logger:
        loggers.append(TerminalLogger(**cfg.logger.terminal))

    # return the loggers
    return EnsembleLogger(loggers)


@hydra.main(
    config_path="config",
    version_base=None,
    config_name="config_exp",
)
def run(cfg: omegaconf.DictConfig) -> None:
    # Check and configure the available devices.
    available_devices = len(jax.local_devices())
    if cfg.num_devices < 0:
        cfg.num_devices = available_devices
        print(f"Using {available_devices} available device(s).")
    else:
        assert (
            available_devices >= cfg.num_devices
        ), f"{cfg.num_devices} devices requested but only {available_devices} available."

    # Create the logger and save the config.
    logger = create_logger(cfg)
    # logger.write_config(cfg)

    # Train!
    trainer = Trainer(cfg, logger)
    trainer.train()

    # Tidy.
    logger.close()


if __name__ == "__main__":
    run()
