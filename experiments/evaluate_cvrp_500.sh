#!/bin/bash

{

#### LOW BUDGET ####
# default 50sp, 500 attempts = 25000
# COMPASS will use 20sp (and latent space 10), hence 125 attempts -> budget 1250
python experiments/slowrl_validate.py --config-name config_exp_cvrp \
    slowrl.environment.num_nodes=500 \
    slowrl.environment.norm_factor=100 \
    slowrl.memory.num_nodes=501 \
    slowrl.memory.memory_size=40 \
    slowrl.num_starting_points=20 \
    slowrl.budget=1250 \
    slowrl.num_cmaes_states=1 \
    slowrl.validation_pop_size=10 \
    slowrl.rollout.policy.temperature=0.3 \
    slowrl.networks.decoder.memory_processing.mlp.num_layers=2 \
    slowrl.networks.decoder.memory_processing.mlp.hidden_size=8 \
    slowrl.problem_seed=0 \
    slowrl.problems.load_problem=true \
    slowrl.problems.num_problems=128 \
    slowrl.problems.load_path="data/validation/vrp500_test_lkh.pkl" \
    slowrl.instances_batch_size=8 \
    slowrl.networks.encoder.query_chunk_size=501 \
    slowrl.networks.decoder.query_chunk_size=501 \
    slowrl.networks.decoder.key_chunk_size=501 \
    slowrl.checkpointing.restore_path="data/memento/checkpoints/cvrp" \
    slowrl.checkpointing.checkpoint_fname_load="compento_500" \
    slowrl.networks.decoder.memory_usage_flags.use_memory=true \
    slowrl.networks.decoder.memory_usage_flags.budget_trick=true \
    slowrl.networks.decoder.memory_usage_flags.remaining_budget_start=0.5 \
    slowrl.networks.decoder.memory_usage_flags.steps_trick=false \


#### HIGH BUDGET ####
# default 100sp, 1000 attempts
# used as for MEMENTO, hence 100sp, budget 1000
python experiments/slowrl_validate.py --config-name config_exp_cvrp \
    slowrl.environment.num_nodes=500 \
    slowrl.environment.norm_factor=100 \
    slowrl.memory.num_nodes=501 \
    slowrl.memory.memory_size=40 \
    slowrl.num_starting_points=40 \
    slowrl.budget=2500 \
    slowrl.num_cmaes_states=1 \
    slowrl.validation_pop_size=10 \
    slowrl.rollout.policy.temperature=0.3 \
    slowrl.networks.decoder.memory_processing.mlp.num_layers=2 \
    slowrl.networks.decoder.memory_processing.mlp.hidden_size=8 \
    slowrl.problem_seed=0 \
    slowrl.problems.load_problem=true \
    slowrl.problems.num_problems=128 \
    slowrl.problems.load_path="data/validation/vrp500_test_lkh.pkl" \
    slowrl.instances_batch_size=8 \
    slowrl.networks.encoder.query_chunk_size=501 \
    slowrl.networks.decoder.query_chunk_size=501 \
    slowrl.networks.decoder.key_chunk_size=501 \
    slowrl.checkpointing.restore_path="data/memento/checkpoints/cvrp" \
    slowrl.checkpointing.checkpoint_fname_load="compento_500" \
    slowrl.networks.decoder.memory_usage_flags.use_memory=true \
    slowrl.networks.decoder.memory_usage_flags.budget_trick=true \
    slowrl.networks.decoder.memory_usage_flags.remaining_budget_start=0.5 \
    slowrl.networks.decoder.memory_usage_flags.steps_trick=false \

}
