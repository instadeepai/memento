#!/bin/bash

{
# 1000 instances of size 125 (loaded)
python experiments/slowrl_validate.py --config-name config_exp_cvrp \
    slowrl.environment.num_nodes=125 \
    slowrl.environment.norm_factor=55 \
    slowrl.num_starting_points=-1 \
    slowrl.problems.load_problem=True \
    slowrl.problems.load_path="data/validation/vrp125_test_small_seed1235.pkl" \
    slowrl.instances_batch_size=25 \
    slowrl.problems.num_problems=1000 \
    slowrl.memory.num_nodes=125 \
    slowrl.memory.memory_size=40 \
    slowrl.budget=1600 \
    slowrl.num_cmaes_states=2 \
    slowrl.validation_pop_size=16 \
    slowrl.networks.decoder.memory_processing.mlp.num_layers=2 \
    slowrl.networks.decoder.memory_processing.mlp.hidden_size=8 \
    slowrl.rollout.policy.temperature=0.1 \
    slowrl.checkpointing.restore_path="data/memento/checkpoints/cvrp" \
    slowrl.networks.decoder.memory_usage_flags.use_memory=true \
    slowrl.networks.decoder.memory_usage_flags.budget_trick=true \
    slowrl.networks.decoder.memory_usage_flags.remaining_budget_start=0.5 \
    slowrl.networks.decoder.memory_usage_flags.steps_trick=false

# 1000 instances of size 150 (loaded)
python experiments/slowrl_validate.py --config-name config_exp_cvrp \
    slowrl.environment.num_nodes=150 \
    slowrl.environment.norm_factor=60 \
    slowrl.num_starting_points=-1 \
    slowrl.problems.load_problem=True \
    slowrl.problems.load_path="data/validation/vrp150_test_small_seed1235.pkl" \
    slowrl.instances_batch_size=25 \
    slowrl.problems.num_problems=1000 \
    slowrl.memory.num_nodes=150 \
    slowrl.memory.memory_size=40 \
    slowrl.budget=1600 \
    slowrl.num_cmaes_states=2 \
    slowrl.validation_pop_size=16 \
    slowrl.networks.decoder.memory_processing.mlp.num_layers=2 \
    slowrl.networks.decoder.memory_processing.mlp.hidden_size=8 \
    slowrl.rollout.policy.temperature=0.1 \
    slowrl.checkpointing.restore_path="data/memento/checkpoints/cvrp" \
    slowrl.networks.decoder.memory_usage_flags.use_memory=true \
    slowrl.networks.decoder.memory_usage_flags.budget_trick=true \
    slowrl.networks.decoder.memory_usage_flags.remaining_budget_start=0.5 \
    slowrl.networks.decoder.memory_usage_flags.steps_trick=false

# 1000 instances of size 200 (loaded)
python experiments/slowrl_validate.py --config-name config_exp_cvrp \
    slowrl.environment.num_nodes=200 \
    slowrl.environment.norm_factor=70 \
    slowrl.num_starting_points=-1 \
    slowrl.problems.load_problem=True \
    slowrl.problems.load_path="data/validation/vrp200_test_small_seed1235.pkl" \
    slowrl.instances_batch_size=25 \
    slowrl.problems.num_problems=1000 \
    slowrl.memory.num_nodes=200 \
    slowrl.memory.memory_size=40 \
    slowrl.budget=1600 \
    slowrl.num_cmaes_states=2 \
    slowrl.validation_pop_size=16 \
    slowrl.networks.encoder.query_chunk_size=201 \
    slowrl.networks.decoder.memory_processing.mlp.num_layers=2 \
    slowrl.networks.decoder.memory_processing.mlp.hidden_size=8 \
    slowrl.rollout.policy.temperature=0.1 \
    slowrl.checkpointing.restore_path="data/memento/checkpoints/cvrp" \
    slowrl.networks.decoder.memory_usage_flags.use_memory=true \
    slowrl.networks.decoder.memory_usage_flags.budget_trick=true \
    slowrl.networks.decoder.memory_usage_flags.remaining_budget_start=0.5 \
    slowrl.networks.decoder.memory_usage_flags.steps_trick=false

# 10000 instances of size 100 (loaded)
python experiments/slowrl_validate.py --config-name config_exp_cvrp \
    slowrl.environment.num_nodes=100 \
    slowrl.environment.norm_factor=50 \
    slowrl.num_starting_points=-1 \
    slowrl.problems.load_problem=True \
    slowrl.problems.load_path="data/validation/vrp100_test_seed1234.pkl" \
    slowrl.instances_batch_size=5 \
    slowrl.problems.num_problems=10000 \
    slowrl.memory.num_nodes=100 \
    slowrl.memory.memory_size=40 \
    slowrl.budget=1600 \
    slowrl.num_cmaes_states=2 \
    slowrl.validation_pop_size=16 \
    slowrl.networks.decoder.memory_processing.mlp.num_layers=2 \
    slowrl.networks.decoder.memory_processing.mlp.hidden_size=8 \
    slowrl.rollout.policy.temperature=0.1 \
    slowrl.checkpointing.restore_path="data/memento/checkpoints/cvrp" \
    slowrl.networks.decoder.memory_usage_flags.use_memory=true \
    slowrl.networks.decoder.memory_usage_flags.budget_trick=true \
    slowrl.networks.decoder.memory_usage_flags.remaining_budget_start=0.5 \
    slowrl.networks.decoder.memory_usage_flags.steps_trick=false

}
