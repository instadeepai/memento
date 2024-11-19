#!/bin/bash

{
#### LOW BUDGET ####
# default 50sp, 500 attempts = 25000
# used as for MEMENTO, hence 50sp, budget 500
python experiments/slowrl_validate.py --config-name config_exp_cvrp \
    slowrl.environment.num_nodes=500 \
    slowrl.environment.norm_factor=100 \
    slowrl.memory.num_nodes=501 \
    slowrl.memory.memory_size=40 \
    slowrl.num_starting_points=50 \
    slowrl.budget=500 \
    slowrl.problems.load_problem=True \
    slowrl.problems.load_path="data/validation/vrp500_test_lkh.pkl" \
    slowrl.batch_size=8 \
    slowrl.problems.num_problems=128 \
    slowrl.networks.decoder.memory_processing.mlp.num_layers=2 \
    slowrl.networks.decoder.memory_processing.mlp.hidden_size=8 \
    slowrl.networks.encoder.query_chunk_size=501 \
    slowrl.networks.decoder.query_chunk_size=500 \
    slowrl.networks.decoder.key_chunk_size=500 \
    slowrl.checkpointing.restore_path="data/memento/checkpoints/cvrp" \
    slowrl.checkpointing.checkpoint_fname_load="memento_500" \
    slowrl.rollout.policy.temperature=0.5

#### HIGH BUDGET ####
# default 100sp, 1000 attempts
# used as for MEMENTO, hence 100sp, budget 1000
python experiments/slowrl_validate.py --config-name config_exp_cvrp \
    slowrl.environment.num_nodes=500 \
    slowrl.environment.norm_factor=100 \
    slowrl.memory.num_nodes=501 \
    slowrl.memory.memory_size=40 \
    slowrl.num_starting_points=100 \
    slowrl.budget=1000 \
    slowrl.problems.load_problem=True \
    slowrl.problems.load_path="data/validation/vrp500_test_lkh.pkl" \
    slowrl.batch_size=8 \
    slowrl.problems.num_problems=128 \
    slowrl.networks.decoder.memory_processing.mlp.num_layers=2 \
    slowrl.networks.decoder.memory_processing.mlp.hidden_size=8 \
    slowrl.networks.encoder.query_chunk_size=501 \
    slowrl.networks.decoder.query_chunk_size=500 \
    slowrl.networks.decoder.key_chunk_size=500 \
    slowrl.checkpointing.restore_path="data/memento/checkpoints/cvrp" \
    slowrl.checkpointing.checkpoint_fname_load="memento_500" \
    slowrl.rollout.policy.temperature=0.5

}
