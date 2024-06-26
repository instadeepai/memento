import hydra
import jax
import omegaconf

from memento.trainers.slowrl_validation import slowrl_validate
from memento.utils.logger import create_logger


@hydra.main(
    config_path="config",
    version_base=None,
    config_name="config_exp",
)
def run(cfg: omegaconf.DictConfig) -> None:

    print(cfg.slowrl.checkpointing.restore_path)
    # Check and configure the available devices.
    behavior_dim = cfg.behavior_dim
    slowrl_cfg = cfg.slowrl

    available_devices = len(jax.local_devices())
    if slowrl_cfg.num_devices < 0:
        slowrl_cfg.num_devices = available_devices
        print(f"Using {available_devices} available device(s).")
    else:
        assert (
            available_devices >= slowrl_cfg.num_devices
        ), f"{slowrl_cfg.num_devices} devices requested but only {available_devices} available."

    # create a logger
    logger = create_logger(cfg)

    # convert percentage exta nodes to num nodes
    if cfg.env_name == "cvrp":
        slowrl_cfg.memory.num_node_buckets = slowrl_cfg.environment.num_nodes + 1
    elif cfg.env_name == "tsp":
        slowrl_cfg.memory.num_node_buckets = slowrl_cfg.environment.num_cities
    else:
        raise ValueError(f"Environment {slowrl_cfg.env_name} not supported")

    print("num node buckets", slowrl_cfg.memory.num_node_buckets)

    # run the validation
    metrics = slowrl_validate(
        cfg=slowrl_cfg, params=None, behavior_dim=behavior_dim, logger=logger
    )

    print(metrics)


if __name__ == "__main__":
    run()
