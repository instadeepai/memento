import hydra
import jax
import omegaconf

from memento.trainers.validation import validate
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
    behavior_dim = cfg.behavior_dim
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
    logger = create_logger(cfg)

    # init the random key
    random_key = jax.random.PRNGKey(validation_cfg.problem_seed)
    metrics = validate(
        random_key=random_key,
        cfg=validation_cfg,
        behavior_dim=behavior_dim,
    )

    metrics = {f"validate/{k}": v for (k, v) in metrics.items()}
    logger.write(metrics)


if __name__ == "__main__":
    run()
