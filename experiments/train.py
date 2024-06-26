from math import ceil

import hydra
import jax
import omegaconf

from memento.trainers.trainer import Trainer
from memento.utils.logger import create_logger


@hydra.main(
    config_path="config",
    version_base=None,
    config_name="config_exp",
)
def run(cfg: omegaconf.DictConfig) -> None:
    # check and configure the available devices.
    available_devices = len(jax.local_devices())
    if cfg.num_devices < 0:
        cfg.num_devices = available_devices
        print(f"Using {available_devices} available device(s).")
    else:
        assert (
            available_devices >= cfg.num_devices
        ), f"{cfg.num_devices} devices requested but only {available_devices} available."

    # create the logger and save the config.
    neptune_tags = ("training",)  # "final-exp")
    logger = create_logger(cfg, tags=neptune_tags)
    # logger.write_config(cfg)

    # convert percentage exta nodes to num nodes
    if cfg.env_name == "cvrp":
        cfg.memory.num_node_buckets = cfg.environment.num_nodes + 1
    elif cfg.env_name == "tsp":
        cfg.memory.num_node_buckets = cfg.environment.num_cities
    else:
        raise ValueError(f"Environment {cfg.env_name} not supported")

    cfg.slowrl.memory.num_node_buckets = cfg.memory.num_node_buckets

    print("num node buckets", cfg.memory.num_node_buckets)

    # train
    trainer = Trainer(cfg, logger)
    trainer.train()

    # tidy.
    logger.close()


if __name__ == "__main__":
    run()
