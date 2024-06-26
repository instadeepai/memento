import chex
from flax import struct


@struct.dataclass
class TSPMemoryDataPoint:
    """Data point stored in the external memory.

    Args:
        position: current position.
        visited_mask: mask of visited nodes.
        action: action taken.
        returns: return of the trajectory.
        logprob: log probability of the action.
    """

    position: chex.Numeric  # ()
    action: chex.Numeric  # ()
    returns: chex.Numeric  # ()
    logprob: chex.Numeric  # ()
    mem_logprob: chex.Numeric  # ()
    attn_logprob: chex.Numeric  # ()
    traj_logprob: chex.Numeric  # ()
    end_traj_logprob: chex.Numeric  # ()
    age: chex.Numeric  # ()


@struct.dataclass
class CVRPMemoryDataPoint:
    """Data point stored in the external memory.

    Args:
        position: current position.
        capacity: current capacity.
        visited_mask: mask of visited nodes.
        action: action taken.
        returns: return of the trajectory.
        logprob: log probability of the action.
    """

    position: chex.Numeric  # ()
    capacity: chex.Numeric  # ()
    action: chex.Numeric  # ()
    returns: chex.Numeric  # ()
    logprob: chex.Numeric  # ()
    mem_logprob: chex.Numeric  # ()
    attn_logprob: chex.Numeric  # ()
    traj_logprob: chex.Numeric  # ()
    end_traj_logprob: chex.Numeric  # ()
    age: chex.Numeric  # ()
