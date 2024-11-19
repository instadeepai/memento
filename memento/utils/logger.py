from abc import ABC, abstractmethod
from typing import Any, List, Mapping, Tuple, Union

import jax
import neptune.new as neptune
import numpy as np
from acme.utils import loggers as acme_loggers

# An artifact is a mapping between a string and the path of a file to log
Path = str
Artifact = Mapping[str, Union[Path, Any]]  # recursive structure


class PoppyLogger(ABC):
    @abstractmethod
    def write(self, data: acme_loggers.LoggingData) -> None:
        pass

    @abstractmethod
    def write_artifact(self, artifact: Artifact) -> None:
        pass

    @abstractmethod
    def close(self):
        pass


class TerminalLogger(acme_loggers.TerminalLogger, PoppyLogger):
    def __init__(self, label: str, time_delta: float, **kwargs: Any):
        super(TerminalLogger, self).__init__(
            label=label, time_delta=time_delta, print_fn=print, **kwargs
        )

    def write_artifact(self, artifact: Artifact) -> None:
        pass


class NeptuneLogger(acme_loggers.Logger, PoppyLogger):
    def __init__(self, **kwargs: Any):
        super(NeptuneLogger, self).__init__()
        self.run = neptune.init(**{k: kwargs[k] for k in ["name", "tags"]})
        self.run["parameters"] = kwargs["parameters"]

    def write(self, data: acme_loggers.LoggingData) -> None:
        for key, value in data.items():
            key_with_label = f"{key}"
            if not np.isscalar(value):
                value = float(value)
            self.run[key_with_label].log(value)

    def write_artifact(self, artifact: Artifact) -> None:
        for key, value in artifact.items():
            self.run[key].upload(value)

    def close(self) -> None:
        self.run.stop()


class EnsembleLogger(acme_loggers.Logger, PoppyLogger):
    def __init__(self, loggers: List):
        self.loggers = loggers

    def write(self, data: acme_loggers.LoggingData) -> None:
        for logger in self.loggers:
            logger.write(data)

    def write_artifact(self, artifact: Artifact) -> None:
        for logger in self.loggers:
            logger.write_artifact(artifact)

    def close(self) -> None:
        for logger in self.loggers:
            logger.close()


def create_logger(cfg, tags: Tuple) -> EnsembleLogger:

    loggers = []

    if "terminal" in cfg.logger:
        loggers.append(TerminalLogger(**cfg.logger.terminal))

    if "neptune" in cfg.logger and jax.process_index() == 0:
        neptune_config = {}
        neptune_config["name"] = cfg.logger.neptune.name
        
        # define tags
        neptune_tags = [f"{cfg.algo_name}", f"{cfg.env_name}"] + list(tags)
        neptune_config["tags"] = neptune_tags

        neptune_config["parameters"] = cfg

        loggers.append(NeptuneLogger(**neptune_config))

    # return the loggers
    return EnsembleLogger(loggers)


def create_subdirectory_path(cfg):
    # create the name of the run's directory - used for logging and checkpoints
    run_subdirectory = (
        str(cfg.env_name)
        + "/"
        + str(cfg.algo_name)
        + "/"
        + str(cfg.subdir_tag)  # additional tag to differentiate runs
        + f"ms{cfg.memory.memory_size}"
        + f"_puf{cfg.budget}"
        + f"_bs{cfg.batch_size}_ps{cfg.pop_size}"
        + f"_ga{cfg.optimizer.num_gradient_accumulation_steps}"
        + f"_seed{cfg.seed}/"
    )

    return run_subdirectory
