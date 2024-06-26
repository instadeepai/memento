#!/bin/bash

{
#### LOW BUDGET ####
# default 50sp, 500 attempts = 25000
python experiments/slowrl_validate.py --config-name config_exp_cvrp \
    slowrl.environment.num_nodes=500 \
    slowrl.environment.norm_factor=100 \
    slowrl.num_starting_points=1 \
    slowrl.budget=5 \
    slowrl.problems.load_problem=True \
    slowrl.problems.load_path="data/validation/vrp500_test_lkh.pkl" \
    slowrl.batch_size=8 \
    slowrl.problems.num_problems=128 \
    slowrl.rollout.policy.temperature=1 \
    slowrl.networks.encoder.query_chunk_size=501 \
    slowrl.networks.decoder.query_chunk_size=501 \
    slowrl.networks.decoder.key_chunk_size=501 \
    slowrl.checkpointing.restore_path="data/memento/checkpoints/cvrp/"


#### HIGH BUDGET ####
# default 100sp, 1000 attempts
python experiments/slowrl_validate.py --config-name config_exp_cvrp \
    slowrl.environment.num_nodes=500 \
    slowrl.environment.norm_factor=100 \
    slowrl.num_starting_points=100 \
    slowrl.budget=1000 \
    slowrl.problems.load_problem=True \
    slowrl.problems.load_path="data/validation/vrp500_test_lkh.pkl" \
    slowrl.batch_size=8 \
    slowrl.problems.num_problems=128 \
    slowrl.rollout.policy.temperature=1 \
    slowrl.networks.encoder.query_chunk_size=501 \
    slowrl.networks.decoder.query_chunk_size=501 \
    slowrl.networks.decoder.key_chunk_size=501 \
    slowrl.checkpointing.restore_path="data/memento/checkpoints/cvrp/"

}