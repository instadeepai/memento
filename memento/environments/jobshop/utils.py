from chex import Array, PRNGKey
from jax import numpy as jnp
from jax import random
import jax


def permute_matrix_columns(key, matrix):
    '''
    Randomly permutes the columns of a matrix using the same permutation for each row.

    matrix: a numpy array

    Returns a permuted version of the input matrix.
    '''
    num_rows, num_cols = matrix.shape
    row_indices = jnp.tile(jnp.arange(num_rows), (num_cols, 1)).T
    col_indices = jax.vmap(lambda k: random.permutation(k, matrix.shape[1]))(random.split(key, num_rows))
    return matrix[row_indices, col_indices]


def generate_problem(problem_key: PRNGKey, num_machines: jnp.int32, num_jobs: jnp.int32, max_ops_duration: jnp.int32, min_ops_duration=1) -> Array:
    '''
    Generates a uniformly random instance of a scheduling problem with given parameters.

    num_jobs: number of jobs to schedule
    num_machines: number of machines available
    time_low: lower bound for job processing times
    time_high: upper bound for job processing times

    Returns a tuple (times, machines) where times is a numpy array of shape (num_jobs, num_machines) containing the
    processing times of each job on each machine, and machines is a numpy array of shape (num_jobs, num_machines)
    specifying the assignment of jobs to machines.
    '''

    duration_key, operation_key = random.split(problem_key)
    durations = random.randint(duration_key, (num_jobs, num_machines), minval=min_ops_duration, maxval=max_ops_duration)
    machines = jnp.tile(jnp.arange(0, num_machines), (num_jobs, 1))
    machines = permute_matrix_columns(operation_key, machines)

    problem = jnp.concatenate((durations.reshape(1, num_jobs, num_machines), machines.reshape(1, num_jobs, num_machines)))

    return problem