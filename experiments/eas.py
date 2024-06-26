import hydra
import jax
import omegaconf

from memento.trainers.eas_emb import eas_emb
from memento.utils.logger import EnsembleLogger, NeptuneLogger, TerminalLogger


def create_logger(cfg) -> EnsembleLogger:
    loggers = []

    if "terminal" in cfg.logger:
        loggers.append(TerminalLogger(**cfg.logger.terminal))

    if "neptune" in cfg.logger:
        neptune_config = {}
        neptune_config["name"] = cfg.logger.neptune.name
        neptune_config["project"] = cfg.logger.neptune.project
        neptune_config["tags"] = [f"{cfg.algo_name}", "eas", f"{cfg.env_name}"]
        neptune_config["parameters"] = cfg

        loggers.append(NeptuneLogger(**neptune_config))

    # return the loggers
    return EnsembleLogger(loggers)


@hydra.main(
    config_path="config",
    version_base=None,
    config_name="config_exp",
)
def run(cfg: omegaconf.DictConfig) -> None:
    # Check and configure the available devices.
    eas_cfg = cfg.eas

    available_devices = len(jax.local_devices())
    if eas_cfg.num_devices < 0:
        eas_cfg.num_devices = available_devices
        print(f"Using {available_devices} available device(s).")
    else:
        assert (
            available_devices >= eas_cfg.num_devices
        ), f"{eas_cfg.num_devices} devices requested but only {available_devices} available."

    # create a logger
    logger = create_logger(cfg)

    # init the random key
    metrics = eas_emb(
        cfg=eas_cfg,
        params=None,
        logger=logger,
    )

    for i in range(eas_cfg.budget):
        metrics_budget_i = {v: metrics[v][i] for v in metrics}
        logger.write(metrics_budget_i)


if __name__ == "__main__":
    run()