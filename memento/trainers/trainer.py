from dataclasses import field
from typing import TYPE_CHECKING, Any, Optional, Tuple

import omegaconf

if TYPE_CHECKING:
    from dataclasses import dataclass

else:
    from chex import dataclass

import functools
import time

import acme
import chex
import haiku as hk
import hydra
import jax
import jax.numpy as jnp
import optax

import memento.trainers.slowrl_validation as slowrl_validation
import memento.trainers.validation as validation
from memento.environments.poppy_env import PoppyEnv
from memento.memory.external_memory import (
    ExternalMemoryState,
    reinitialize_memory,
    update_memory,
)
from memento.networks import Networks
from memento.utils.acting_utils import generate_trajectory
from memento.utils.checkpoint import (
    create_checkpoint_directory,
    load_checkpoint,
    load_memory_decoder_checkpoint,
    save_checkpoint,
)
from memento.utils.data import generate_zeros_from_spec, prepare_problem_batch
from memento.utils.devices import (
    fetch_from_first_device,
    reduce_from_devices,
    run_on_master,
    spread_over_devices,
)
from memento.utils.loss import get_loss_fn, get_rectified_sum_weights
from memento.utils.networks import get_networks


@dataclass
class TrainingState:  # type: ignore
    """Container for data used during the acting in the environment."""

    # TODO: add the latent vectors
    params: hk.Params
    behavior_markers: chex.Array
    optimizer_state: optax.OptState
    num_steps: jnp.int32
    key: chex.PRNGKey
    problem_key: chex.PRNGKey
    memory_state: ExternalMemoryState
    extras: Optional[dict] = field(default_factory=dict)


def get_optimizer(cfg: omegaconf.DictConfig) -> optax.GradientTransformation:
    encoder_mask_fn = functools.partial(
        hk.data_structures.map, lambda m, n, p: "encoder" in m
    )
    decoder_mask_fn = functools.partial(
        hk.data_structures.map,
        lambda m, n, p: ("encoder" not in m)
        and ("memory" not in m)
        and ("memory" not in n),
    )
    memory_mask_fn = functools.partial(
        hk.data_structures.map,
        lambda m, n, p: (("memory" in m) or ("memory" in n)),
    )

    optimizer = optax.chain(
        optax.masked(
            optax.adamw(
                learning_rate=cfg.encoder.lr,
                weight_decay=cfg.encoder.l2_regularization,
            ),
            encoder_mask_fn,
        ),
        optax.masked(
            optax.adamw(
                learning_rate=cfg.decoder.lr,
                weight_decay=cfg.decoder.l2_regularization,
            ),
            decoder_mask_fn,
        ),
        optax.masked(
            optax.adamw(
                learning_rate=cfg.memory.lr,
                weight_decay=cfg.memory.l2_regularization,
            ),
            memory_mask_fn,
        ),
    )
    optimizer = optax.MultiSteps(optimizer, cfg.num_gradient_accumulation_steps)

    return optimizer


def init_training_state(
    cfg: omegaconf.DictConfig,
    networks: Networks,
    environment: PoppyEnv,
    first_rectified_sum_weight: jnp.float32,
) -> TrainingState:
    """Initialise the training state.

    Load checkpoints.
    Manipulate the decoder parameters.
    Create the external memory.
    Create optimizer parameters.


    """

    # define the initial key
    key = jax.random.PRNGKey(cfg.seed)

    # split it in three new keys
    encoder_key, decoder_key, training_key = jax.random.split(key, 3)

    # print("Config loss - type: ", cfg.loss.type)
    # print("Config loss - sp specialisation: ", cfg.loss.sp_spec)

    (
        encoder_params,
        loaded_conditioned_decoder_params,
        optimizer_state,
        keys,
        problem_key,
        num_steps,
        extras,
    ) = load_checkpoint(cfg)

    loaded_memory_decoder_params = load_memory_decoder_checkpoint(cfg)

    if cfg.loss.sp_spec:
        extras["best_return"] = jnp.zeros(
            (
                cfg.batch_size // cfg.num_devices,
                # cfg.training_sample_size,
            )
        )
    else:
        extras["best_return"] = jnp.zeros(
            (
                cfg.batch_size // cfg.num_devices,
                # cfg.training_sample_size,
                cfg.num_starting_positions,
            )
        )

    extras["rectified_sum_weight"] = first_rectified_sum_weight
    extras["first_step"] = True

    # create a dummy observation
    environment_spec = acme.make_environment_spec(environment)
    _dummy_obs = environment.make_observation(
        *jax.tree_util.tree_map(
            generate_zeros_from_spec,
            environment_spec.observations.generate_value(),
        )
    )

    # TODO: node-sp specific memory
    external_memory = hydra.utils.instantiate(cfg.memory)
    memory_state = external_memory.init_state(
        budget=cfg.budget * cfg.training_sample_size
    )

    # replicate the memory to have on for each node
    memory_state = jax.tree_map(
        lambda x: jnp.repeat(x[None, ...], repeats=cfg.memory.num_node_buckets, axis=0),
        memory_state,
    )

    # init encoder params if not loaded - in MEMO, we expect to load params from POMO
    if not encoder_params:
        encoder_params = networks.encoder_fn.init(encoder_key, _dummy_obs.problem)

    # init the decoder params
    _dummy_embeddings = networks.encoder_fn.apply(encoder_params, _dummy_obs.problem)

    _dummy_behavior_marker = jnp.zeros(shape=(cfg.behavior_dim))

    # init conditioned_decoder_params
    decoder_params = networks.decoder_fn.init(
        decoder_key,
        _dummy_obs,
        _dummy_embeddings,
        _dummy_behavior_marker,
        memory_state,
    )

    if loaded_conditioned_decoder_params is not None:

        # parameters need to be merged, not replace
        decoder_params = hk.data_structures.merge(
            decoder_params, loaded_conditioned_decoder_params
        )

        # decrease the typical scale of the memory decoder params
        decrease_scale_fn = lambda module_name, name, value: (
            cfg.init_mem_mha_scale * value if "memory" in module_name else value
        )

        decoder_params = hk.data_structures.map(decrease_scale_fn, decoder_params)
    else:
        # throw an error
        raise ValueError("No decoder params loaded")

    if loaded_memory_decoder_params is not None:
        # let's split the loaded memory decoder params
        loaded_memory_processing_params, _other_decoder_params = (
            hk.data_structures.partition(
                lambda m, n, p: (("memory" in m) or ("memory" in n)),
                loaded_memory_decoder_params,
            )
        )

        if False:
            # remove the first dimension
            loaded_memory_processing_params = jax.tree_map(
                lambda x: x.squeeze(0), loaded_memory_processing_params
            )
        else:
            # extract the first element (it was a pop)
            loaded_memory_processing_params = jax.tree_map(
                lambda x: x[0], loaded_memory_processing_params
            )

        # now merge the params
        decoder_params = hk.data_structures.merge(
            decoder_params, loaded_memory_processing_params
        )

    if not keys:
        training_key, problem_key = jax.random.split(training_key, 2)
        keys = list(jax.random.split(training_key, cfg.num_devices))

    problem_keys = list(jax.random.split(problem_key, cfg.num_devices))

    # distribute parameters over devices as required
    devices = jax.local_devices()
    encoder_params = jax.device_put_replicated(encoder_params, devices)

    # decoding is parallelised over the batch --> every agent needs to be on every device.
    decoder_params = jax.device_put_replicated(decoder_params, devices)

    # merge encoder and decoder params in the same structure
    params = hk.data_structures.merge(encoder_params, decoder_params)

    # define the behavior markers
    behavior_markers = jnp.zeros((cfg.training_sample_size, cfg.behavior_dim))

    if not optimizer_state:
        optimizer_state = get_optimizer(cfg.optimizer).init(
            fetch_from_first_device(params)
        )

    # Replicate memory to have one for each start position

    memory_state = jax.tree_map(
        lambda x: jnp.repeat(x[None, ...], repeats=cfg.num_starting_positions, axis=0),
        memory_state,
    )

    memory_state = jax.tree_map(
        lambda x: jnp.repeat(x[None, ...], repeats=cfg.batch_size, axis=0),
        memory_state,
    )

    memory_state = spread_over_devices(memory_state)

    training_state = TrainingState(
        params=params,
        behavior_markers=jax.device_put_replicated(behavior_markers, devices),
        optimizer_state=jax.device_put_replicated(optimizer_state, devices),
        num_steps=jax.device_put_replicated(num_steps, devices),
        key=jax.device_put_sharded(keys, devices),
        problem_key=jax.device_put_sharded(problem_keys, devices),
        memory_state=memory_state,
        extras=jax.device_put_replicated(extras, devices),
    )

    return training_state


def rollout(
    cfg: omegaconf.DictConfig,
    environment: PoppyEnv,
    params: chex.ArrayTree,
    behavior_markers: chex.Array,
    networks: Networks,
    problems: jnp.ndarray,
    start_positions: jnp.ndarray,
    acting_keys: jnp.ndarray,
    memory_state: Optional[chex.ArrayTree],
) -> Tuple[Any, Tuple[Any, Any, Any]]:
    """Rollout a batch of agents on a batch of problems and starting points.

    Args:
        cfg: The rollout config.
        environment: The environment to rollout.
        params: Dictionary of parameters for all Networks.  Encoder params are assumed to be shared
          across all agents, decoder params are assumed to have a leading dimension of shape K.
        behavior_markers: latent vectors
        networks: The required networks.
        problems: A batch of N problems ([N, problem_size, 2]).
        start_positions: M starting positions for each problem-agent pair ([N, K, M]).
        acting_keys: M acting keys for each problem-agent pair ([N, K, M, 2]).
        memory_state: External memory to be used by the decoder.


    """

    # initialise the embeddings for each problem
    encoder_params, decoder_params = hk.data_structures.partition(
        lambda m, n, p: "encoder" in m, params
    )

    # get the embeddings for each problem - done once
    embeddings = jax.vmap(networks.encoder_fn.apply, in_axes=(None, 0))(
        encoder_params, problems
    )

    @functools.partial(jax.vmap, in_axes=(0, 0, None, None, 0, 0, 0))  # over N problems
    @functools.partial(
        jax.vmap, in_axes=(None, None, None, 0, 0, 0, None)
    )  # over K agents
    @functools.partial(
        jax.vmap, in_axes=(None, None, None, None, 0, 0, 0)
    )  # M starting pos.
    def generate_trajectory_fn(
        problem,
        embeddings,
        decoder_params,
        behavior_markers,
        start_position,
        acting_key,
        memory_state,
    ):

        return generate_trajectory(
            networks.decoder_fn.apply,
            cfg.rollout.policy.temperature,
            environment,
            problem,
            embeddings,
            decoder_params,
            behavior_markers,
            start_position,
            acting_key,
            memory_state,
        )

    # generate the traj
    acting_state, (traj, info) = generate_trajectory_fn(
        problems,
        embeddings,
        decoder_params,
        behavior_markers,
        start_positions,
        acting_keys,
        memory_state,
    )

    info.metrics = jax.tree_map(lambda x: x.mean(), info.metrics)

    # compute the memory entries
    external_memory = hydra.utils.instantiate(cfg.memory)
    data = external_memory.entries_from_trajs(traj, info, embeddings)

    return acting_state, (traj, info, data)


class Trainer:
    def __init__(
        self,
        cfg: omegaconf.DictConfig,
        logger,
    ):
        if cfg.num_devices < 0:
            cfg.num_devices = len(jax.local_devices())

        # define attributes
        self.cfg = cfg
        self.logger = logger
        self.environment = hydra.utils.instantiate(cfg.environment)
        self.networks = get_networks(cfg.networks)

        # loss
        self.compute_loss = get_loss_fn(cfg.loss)
        self.rectified_sum_weights = get_rectified_sum_weights(cfg)

        # create a directory
        create_checkpoint_directory(cfg, self.logger)

        # definie the initial training state
        self.training_state = init_training_state(
            cfg, self.networks, self.environment, self.rectified_sum_weights[0]
        )

        # propagate the num devices to the validation config
        self.cfg.validation.num_devices = self.cfg.num_devices

        def sgd_step(training_state):
            def loss_and_output(
                params, behavior_markers, problems, start_positions, acting_keys
            ):
                # rollout the agents on the problems
                state, (traj, info, data) = rollout(
                    self.cfg,
                    self.environment,
                    params,
                    behavior_markers,
                    self.networks,
                    problems,
                    start_positions,
                    acting_keys,
                    training_state.memory_state,
                )

                # compute the loss on the traj
                loss = self.compute_loss(
                    traj,
                    info,
                    training_state.extras,
                )

                # log loss and returns.
                info.metrics["loss"] = loss

                episode_return = traj.reward.sum(-1)  # [N, K, M]

                if self.environment.is_reward_negative():
                    ret_sign = -1
                else:
                    ret_sign = 1
                return_str = self.environment.get_reward_string()

                info.metrics[f"{return_str}"] = (
                    ret_sign * episode_return.max((-1, -2)).mean()
                )
                if self.cfg.training_sample_size > 1:
                    info.metrics[f"{return_str}_rand_agent"] = (
                        ret_sign * episode_return.max(-1).mean()
                    )
                if self.cfg.num_starting_positions != 1:
                    info.metrics[f"{return_str}_rand_start"] = (
                        ret_sign * episode_return.max(-2).mean()
                    )
                if (self.cfg.training_sample_size > 1) and (
                    self.cfg.num_starting_positions != 1
                ):
                    info.metrics[f"{return_str}_rand_agent+start"] = (
                        ret_sign * episode_return.mean()
                    )

                return loss, (state, (traj, info, data))

            # get keys
            key, sp_act_key = jax.random.split(training_state.key)

            num_problems = self.cfg.batch_size // self.cfg.num_devices

            num_agents = self.cfg.training_sample_size

            # prepare batch of problems, start positions and acting keys
            problems, start_positions, acting_keys = prepare_problem_batch(
                problem_key=training_state.problem_key,
                start_act_key=sp_act_key,
                environment=self.environment,
                num_problems=num_problems,
                num_agents=num_agents,
                num_start_positions=self.cfg.num_starting_positions,
            )

            params = training_state.params
            optimizer_state = training_state.optimizer_state

            # sample behavior markers
            # WARNING: i need to make sure that the key is the same
            # while the problems are the same!
            # hence, i'll be using the problem key
            behavior_markers = cfg.behavior_amplification * jax.random.uniform(
                training_state.problem_key,
                shape=training_state.behavior_markers.shape,
                minval=-1,
                maxval=1,
            )

            # compute the grads wrt the params (encoder + decoder)
            grads, (_state, (traj, info, data)) = jax.grad(
                loss_and_output,
                has_aux=True,
            )(
                params,
                behavior_markers,
                problems,
                start_positions,
                acting_keys,
            )

            if self.cfg.num_devices > 1:
                # Taking the mean across all devices to keep params in sync.
                grads = jax.lax.pmean(grads, axis_name="i")

            # get updates to be applied to params
            updates, optimizer_state = get_optimizer(self.cfg.optimizer).update(
                grads, optimizer_state, params=params
            )

            # apply the update (backpropagation)
            params = optax.apply_updates(params, updates)

            data = jax.tree_map(
                lambda x: x.transpose((0, 2, 3, 1, 4)), data
            )  # [pbs, bd_dim, sp, nodes, feat_dim] -> [pbs, sp, nodes, bd_dim, feat_dim]

            memory_state = jax.vmap(jax.vmap(jax.vmap(update_memory)))(
                training_state.memory_state,
                data,
            )

            # get the best return and update the overall best for each problem
            episode_return = traj.reward.sum(-1)  # [N, K, M]

            if self.cfg.loss.sp_spec:
                best_return = episode_return.max((-2, -1))  # [N, K, M] -> [N]
                # best_return = episode_return.max(-1)  # [N, K, M] -> [N, K]
            else:
                best_return = episode_return.max(-2)  # [N, K, M] -> [N, M]
                # best_return = episode_return  # [N, K, M] -> [N, K, M]

            # update the extras dictionary in training state
            condition = training_state.extras["best_return"] == 0

            if self.cfg.loss.sp_spec:
                training_state.extras["best_return"] = best_return * (condition) + (
                    1 - condition
                ) * jnp.maximum(training_state.extras["best_return"], best_return)
            else:

                # print shapes
                print("Best return shape : ", best_return.shape)
                print(
                    "Stored best return : ", training_state.extras["best_return"].shape
                )

                updated_best_return = jnp.where(
                    condition,
                    best_return,
                    jnp.maximum(training_state.extras["best_return"], best_return),
                )
                training_state.extras["best_return"] = updated_best_return

                print("New best return : ", updated_best_return.shape)

            training_state.extras["rectified_sum_weight"] = self.rectified_sum_weights[
                # (fetch_from_first_device(training_state.num_steps) + 1)
                (training_state.num_steps[0] + 1)
                % self.cfg.budget
            ]

            # set first step to False
            training_state.extras["first_step"] = False

            training_state = TrainingState(
                params=params,
                behavior_markers=behavior_markers,
                optimizer_state=optimizer_state,
                key=key,
                problem_key=training_state.problem_key,  # keep the same problem key
                memory_state=memory_state,
                num_steps=training_state.num_steps + 1,
                extras=training_state.extras,
            )

            return training_state, info.metrics

        @functools.partial(jax.pmap, axis_name="i")
        def n_sgd_steps(training_state):
            # apply n steps of sgd
            training_state, metrics = jax.lax.scan(
                lambda state, _xs: sgd_step(state),
                init=training_state,
                xs=None,
                length=self.cfg.num_jit_steps,
            )

            # average metrics over all jit-ted steps.
            metrics = jax.tree_map(lambda x: x.mean(0), metrics)

            return training_state, metrics

        self.n_sgd_steps = n_sgd_steps

    def train(self):  # noqa: CCR001
        def get_n_steps():
            if self.cfg.num_devices > 1:
                n_steps = fetch_from_first_device(self.training_state.num_steps)
            else:
                n_steps = self.training_state.num_steps
            return n_steps

        @run_on_master
        def log(metrics, key=None):
            metrics["step"] = get_n_steps()
            if self.logger:
                if key:
                    metrics = {f"{key}/{k}": v for (k, v) in metrics.items()}

                self.logger.write(metrics)

        # main while loop
        while get_n_steps() <= self.cfg.num_steps:
            # validation step - happening every validation_freq steps
            if get_n_steps() % self.cfg.validation_freq == 0:
                # fetch the training state
                training_state = fetch_from_first_device(self.training_state)

                # get slow RL validation metrics
                t = time.time()

                # use a fixed key to always get same setting
                metrics = slowrl_validation.slowrl_validate(
                    self.cfg.slowrl,
                    training_state.params,
                    behavior_dim=self.cfg.behavior_dim,
                    logger=self.logger,
                )

                metrics["total_time"] = time.time() - t
                if self.cfg.num_devices > 1:
                    metrics = reduce_from_devices(metrics, axis=0)

                log(metrics, "slowrl_validate")

                reward_str = self.environment.get_reward_string()
                if self.cfg.checkpointing.save_checkpoint:
                    training_state = fetch_from_first_device(
                        self.training_state
                    ).replace(key=self.training_state.key)
                    save_checkpoint(
                        self.cfg,
                        training_state,
                        self.logger,
                    )

                    if (
                        metrics[reward_str] > training_state.extras["best_reward"]
                        and self.cfg.checkpointing.keep_best_checkpoint
                    ):
                        save_checkpoint(
                            self.cfg,
                            training_state,
                            self.logger,
                            fname_prefix="best_",
                        )

                        extras = self.training_state.extras
                        extras.update(
                            {
                                "best_reward": jnp.ones_like(extras["best_reward"])
                                * metrics[reward_str]
                            }
                        )

                        self.training_state = TrainingState(
                            params=self.training_state.params,
                            behavior_markers=self.training_state.behavior_markers,
                            optimizer_state=self.training_state.optimizer_state,
                            num_steps=self.training_state.num_steps,
                            key=self.training_state.key,
                            problem_key=self.training_state.problem_key,
                            memory_state=self.training_state.memory_state,
                            extras=extras,
                        )

                    print(f"Saved checkpoint at step {get_n_steps()}")

            t = time.time()

            # main training step
            self.training_state, metrics = self.n_sgd_steps(self.training_state)

            if get_n_steps() % self.cfg.budget == 0:
                # update the problem key in training state
                new_problem_keys = jax.pmap(
                    # TODO: not sure why I split this with num_devices
                    lambda x: jax.random.split(x, self.cfg.num_devices)[1],
                    axis_name="i",
                )(self.training_state.problem_key)

                # reinit the memory
                new_memory = jax.pmap(
                    reinitialize_memory,
                    axis_name="i",
                )(self.training_state.memory_state)

                # update the extra best return to zero
                new_extras = self.training_state.extras
                new_extras["best_return"] = jnp.zeros_like(
                    self.training_state.extras["best_return"]
                )
                new_extras["first_step"] = jnp.ones_like(
                    self.training_state.extras["first_step"]
                )

                # update the training state
                self.training_state = self.training_state.replace(
                    problem_key=new_problem_keys,
                    memory_state=new_memory,
                    extras=new_extras,
                )

            jax.tree_map(
                lambda x: x.block_until_ready(), metrics
            )  # For accurate timings.

            if self.cfg.num_devices > 1:
                metrics = reduce_from_devices(metrics, axis=0)

            # compute the training (& optionally validate) step time
            metrics["step_time"] = (time.time() - t) / self.cfg.num_jit_steps

            log(metrics, "train")
