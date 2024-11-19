#!/bin/bash

{
###### VALIDATION ########
# 1000 instances of size 125 (loaded)
python experiments/slowrl_validate.py \
    slowrl.environment.num_cities=125 \
    slowrl.num_starting_points=2 \
    slowrl.problems.load_problem=True \
    slowrl.problems.load_path="data/validation/tsp125_test_small_seed1235.pkl" \
    slowrl.instances_batch_size=25 \
    slowrl.problems.num_problems=1000 \
    slowrl.memory.num_nodes=125 \
    slowrl.memory.memory_size=40 \
    slowrl.budget=2 \
    slowrl.networks.decoder.memory_processing.mlp.num_layers=2 \
    slowrl.networks.decoder.memory_processing.mlp.hidden_size=8 \
    slowrl.checkpointing.restore_path="data/memento/checkpoints/tsp" \
    slowrl.checkpointing.checkpoint_fname_load="memento_100" \
    slowrl.rollout.policy.temperature=0.2 \

# 1000 instances of size 150 (loaded)
python experiments/slowrl_validate.py \
    slowrl.environment.num_cities=150 \
    slowrl.num_starting_points=-1 \
    slowrl.problems.load_problem=True \
    slowrl.problems.load_path="data/validation/tsp150_test_small_seed1235.pkl" \
    slowrl.instances_batch_size=25 \
    slowrl.problems.num_problems=1000 \
    slowrl.memory.num_nodes=150 \
    slowrl.memory.memory_size=40 \
    slowrl.budget=1600 \
    slowrl.networks.decoder.memory_processing.mlp.num_layers=2 \
    slowrl.networks.decoder.memory_processing.mlp.hidden_size=8 \
    slowrl.checkpointing.restore_path="data/memento/checkpoints/tsp" \
    slowrl.checkpointing.checkpoint_fname_load="memento_100" \
    slowrl.rollout.policy.temperature=0.2 \

# 1000 instances of size 200 (loaded)
python experiments/slowrl_validate.py \
    slowrl.environment.num_cities=200 \
    slowrl.num_starting_points=-1 \
    slowrl.problems.load_problem=True \
    slowrl.problems.load_path="data/validation/tsp200_test_small_seed1235.pkl" \
    slowrl.instances_batch_size=25 \
    slowrl.problems.num_problems=1000 \
    slowrl.memory.num_nodes=200 \
    slowrl.memory.memory_size=40 \
    slowrl.budget=1600 \
    slowrl.networks.decoder.memory_processing.mlp.num_layers=2 \
    slowrl.networks.decoder.memory_processing.mlp.hidden_size=8 \
    slowrl.checkpointing.restore_path="data/memento/checkpoints/tsp" \
    slowrl.checkpointing.checkpoint_fname_load="memento_100" \
    slowrl.rollout.policy.temperature=0.1 \

# 10000 instances of size 100 (loaded)
python experiments/slowrl_validate.py \
    slowrl.environment.num_cities=100 \
    slowrl.num_starting_points=-1 \
    slowrl.problems.load_problem=True \
    slowrl.problems.load_path="data/validation/tsp100_test_seed1234.pkl" \
    slowrl.instances_batch_size=125 \
    slowrl.problems.num_problems=10000 \
    slowrl.memory.num_nodes=100 \
    slowrl.memory.memory_size=40 \
    slowrl.budget=1600 \
    slowrl.networks.decoder.memory_processing.mlp.num_layers=2 \
    slowrl.networks.decoder.memory_processing.mlp.hidden_size=8 \
    slowrl.checkpointing.restore_path="data/memento/checkpoints/tsp" \
    slowrl.checkpointing.checkpoint_fname_load="memento_100" \
    slowrl.rollout.policy.temperature=1 \

}