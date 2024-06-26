from dataclasses import field
from typing import TYPE_CHECKING, Dict, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import rlax
from chex import Array, PRNGKey
from jumanji.environments.packing.knapsack.types import State as StateKnapsack
from jumanji.environments.routing.cvrp.types import State as StateCVRP
from jumanji.environments.routing.tsp.types import State as StateTSP
from jumanji.types import TimeStep

from memento.environments.cvrp.types import Observation as ObservationCVRP
from memento.environments.knapsack.types import Observation as ObservationKnapsack
from memento.environments.tsp.types import Observation as ObservationTSP

if TYPE_CHECKING:
    from dataclasses import dataclass

else:
    from chex import dataclass

State = Union[StateTSP, StateKnapsack, StateCVRP]
Observation = Union[ObservationTSP, ObservationKnapsack, ObservationCVRP]


@dataclass
class ActingState:  # type: ignore
    """Container for data used during the acting in the environment."""

    state: State
    timestep: TimeStep
    key: PRNGKey


@dataclass
class Information:  # type: ignore
    extras: Optional[dict] = field(default_factory=dict)
    metrics: Optional[dict] = field(default_factory=dict)
    logging: Optional[dict] = field(default_factory=dict)


def true_depot_count(state):
    return state.num_total_visits - 1 - jnp.count_nonzero(state.trajectory)


def dummy_depot_count(state):
    return 0


def generate_trajectory(
    decoder_apply_fn,
    policy_temperature,
    environment,
    problem,
    embeddings,
    params,
    behavior_marker,
    start_position,
    acting_key,
    memory_state,
):
    """Decode a single agent, from a single starting position on a single problem.

    With decorators, the expected input dimensions are:
        problems: [N, problem_size, 2]
        embeddings: [N, problem_size, 128]
        params (decoder only): {key: [K, ...]}
        start_position: [N, K, M]
        acting_key: [N, K, M, 2]
        external_memory: {"key": [N, memory_size, key_dim], "value": [N, memory_size, value_dim]}
    """

    def policy(
        key: PRNGKey,
        observation: Observation,
    ) -> Tuple[Array, Array, Array, Array, Array, Dict]:
        # get logits from the decoder
        logits, attn_logits, mem_logits, intermediate_context, retrieval_metrics = (
            decoder_apply_fn(
                params, observation, embeddings, behavior_marker, memory_state
            )
        )

        # apply mask to the logits
        logits -= 1e30 * observation.action_mask

        # take the action
        if policy_temperature > 0:
            action = rlax.softmax(temperature=policy_temperature).sample(key, logits)
        else:
            action = rlax.greedy().sample(key, logits)

        # calculate the logprob
        logprob = rlax.softmax(temperature=policy_temperature).logprob(
            sample=action, logits=logits
        )

        attn_logprob = rlax.softmax(temperature=policy_temperature).logprob(
            sample=action, logits=attn_logits
        )
        mem_logprob = rlax.softmax(temperature=policy_temperature).logprob(
            sample=action, logits=mem_logits
        )

        return (
            action,
            logprob,
            attn_logprob,
            mem_logprob,
            intermediate_context,
            retrieval_metrics,
        )

    def take_step(acting_state: ActingState):
        # split keys
        key, act_key = jax.random.split(acting_state.key, 2)

        # call the policy
        action, logprob, attn_logprob, mem_logprob, intermediate_context, metrics = (
            policy(act_key, acting_state.timestep.observation)
        )

        # take a step in the environment
        state, timestep = environment.step(acting_state.state, action)
        extras = {
            "logprob": logprob,
            "attn_logprob": attn_logprob,
            "mem_logprob": mem_logprob,
            "action": action,
            # "intermediate_context": intermediate_context,
            "current_node": acting_state.timestep.observation.position,
            # "visited_mask": acting_state.state.visited_mask,
            "current_budget": memory_state.current_budget[0],
            "budget": memory_state.budget[0],
            "behavior_marker": behavior_marker,
        }
        if hasattr(memory_state.data, "capacity"):
            extras["capacity"] = acting_state.timestep.observation.capacity

        # store the logprob for the REINFORCE update
        info = Information(
            extras=extras,
            metrics=metrics,
            logging={},
        )

        # update the acting state
        acting_state = ActingState(state=state, timestep=timestep, key=key)

        return acting_state, (timestep, info)

    state, timestep = environment.reset_from_state(problem, start_position)

    acting_state = ActingState(state=state, timestep=timestep, key=acting_key)

    acting_state, (traj, info) = jax.lax.scan(
        lambda acting_state, _: take_step(acting_state),
        acting_state,
        xs=None,
        length=environment.get_episode_horizon(),
    )

    # Mask logprob's for steps where the environement was done.
    #  - traj.observation.is_done = [0,0,...,0,1,1,...] with the first 1 at the terminal step.
    #  - we want to mask everything *after* the last step, hence the roll & setting the
    #    first step to always (obviously) not be done.
    # TODO: should this be done inside of the rollout function by default?
    is_done = jnp.roll(traj.observation.is_done, 1, axis=-1).at[..., 0].set(0)
    info.extras["logprob"] *= 1 - is_done

    return acting_state, (traj, info)
