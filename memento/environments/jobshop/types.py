from typing import TYPE_CHECKING, NamedTuple, Optional

if TYPE_CHECKING:  # https://github.com/python/mypy/issues/6239
    from dataclasses import dataclass
else:
    from chex import dataclass

import jax.numpy as jnp
import chex


@dataclass
class State:
    """The environment state containing a complete description of the job shop scheduling problem.
    ops_machine_ids: for each job, it specifies the machine each op must be processed on.
        Note that a -1 corresponds to padded ops since not all jobs have the same number of ops.
    ops_durations: for each job, it specifies the processing time of each operation.
        Note that a -1 corresponds to padded ops since not all jobs have the same number of ops.
    ops_mask: for each job, indicates which operations remain to be scheduled. False if the
        op has been scheduled or if the op was added for padding, True otherwise. The first True in
        each row (i.e. each job) identifies the next operation for that job.
    machines_job_ids: for each machine, it specifies the job currently being processed. Note that
        the index num_jobs represents a no-op for which the time until available is always 0.
    machines_remaining_times: for each machine, it specifies the number of time steps until
        available.
    action_mask: for each machine, it indicates which jobs (or no-op) can legally be scheduled.
        The last column corresponds to no-op.
    step_count: used to track time, which is necessary for updating scheduled_times.
    scheduled_times: for each job, it specifies the time at which each operation was scheduled.
        Note that -1 means the operation has not been scheduled yet.
    key: random key used for auto-reset.
    """

    ops_machine_ids: chex.Array  # (num_jobs, max_num_ops)
    ops_durations: chex.Array  # (num_jobs, max_num_ops)
    ops_mask: chex.Array  # (num_jobs, max_num_ops)
    machines_job_ids: chex.Array  # (num_machines,)
    machines_remaining_times: chex.Array  # (num_machines,)
    action_mask: Optional[chex.Array]  # (num_machines, num_jobs + 1)
    step_count: chex.Numeric  # ()
    scheduled_times: chex.Array  # (num_jobs, max_num_ops)
    key: chex.PRNGKey  # (2,)
    is_done: jnp.int32

class Observation(NamedTuple):
    """
    ops_machine_ids: for each job, it specifies the machine each op must be processed on.
        Note that a -1 corresponds to padded ops since not all jobs have the same number of ops.
    ops_durations: for each job, it specifies the processing time of each operation.
        Note that a -1 corresponds to padded ops since not all jobs have the same number of ops.
    ops_mask: for each job, indicates which operations remain to be scheduled. False if the
        op has been scheduled or if the op was added for padding, True otherwise. The first True in
        each row (i.e. each job) identifies the next operation for that job.
    machines_job_ids: for each machine, it specifies the job currently being processed. Note that
        the index num_jobs represents a no-op for which the time until available is always 0.
    machines_remaining_times: for each machine, it specifies the number of time steps until
        available.
    action_mask: for each machine, it indicates which jobs (or no-op) can legally be scheduled.
        The last column corresponds to no-op.
    """

    problem: chex.Array  # (2, num_jobs, num_machines)
    ops_mask: chex.Array  # (num_jobs, max_num_ops)
    machines_job_ids: chex.Array  # (num_machines,)
    machines_remaining_times: chex.Array  # (num_machines,)
    action_mask: chex.Array  # (num_machines, num_jobs + 1)
    is_done: jnp.int32