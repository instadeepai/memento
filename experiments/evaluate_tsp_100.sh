
#!/bin/bash

{

# 1000 instances of size 125 (loaded)
python experiments/slowrl_validate.py \
    slowrl.environment.num_cities=125 \
    slowrl.memory.num_nodes=125 \
    slowrl.memory.memory_size=80 \
    slowrl.num_starting_points=-1 \
    slowrl.budget=1600 \
    slowrl.num_cmaes_states=3 \
    slowrl.validation_pop_size=16 \
    slowrl.rollout.policy.temperature=0.1 \
    slowrl.networks.decoder.memory_processing.mlp.num_layers=2 \
    slowrl.networks.decoder.memory_processing.mlp.hidden_size=8 \
    slowrl.problem_seed=0 \
    slowrl.problems.load_problem=True \
    slowrl.problems.load_path="data/validation/tsp125_test_small_seed1235.pkl" \
    slowrl.instances_batch_size=25 \
    slowrl.problems.num_problems=1000 \
    slowrl.networks.encoder.query_chunk_size=125 \
    slowrl.networks.decoder.query_chunk_size=125 \
    slowrl.networks.decoder.key_chunk_size=125 \
    slowrl.checkpointing.restore_path="data/memento/checkpoints/tsp" \
    slowrl.networks.decoder.memory_usage_flags.use_memory=true \
    slowrl.networks.decoder.memory_usage_flags.budget_trick=true \
    slowrl.networks.decoder.memory_usage_flags.remaining_budget_start=0.2 \
    slowrl.networks.decoder.memory_usage_flags.steps_trick=false \

# 1000 instances of size 150 (loaded)
python experiments/slowrl_validate.py \
    slowrl.environment.num_cities=150 \
    slowrl.memory.num_nodes=150 \
    slowrl.memory.memory_size=80 \
    slowrl.num_starting_points=-1 \
    slowrl.budget=1600 \
    slowrl.num_cmaes_states=3 \
    slowrl.validation_pop_size=16 \
    slowrl.rollout.policy.temperature=0.1 \
    slowrl.networks.decoder.memory_processing.mlp.num_layers=2 \
    slowrl.networks.decoder.memory_processing.mlp.hidden_size=8 \
    slowrl.problem_seed=0 \
    slowrl.problems.load_problem=True \
    slowrl.problems.load_path="data/validation/tsp150_test_small_seed1235.pkl" \
    slowrl.instances_batch_size=25 \
    slowrl.problems.num_problems=1000 \
    slowrl.networks.encoder.query_chunk_size=150 \
    slowrl.networks.decoder.query_chunk_size=150 \
    slowrl.networks.decoder.key_chunk_size=150 \
    slowrl.checkpointing.restore_path="data/memento/checkpoints/tsp" \
    slowrl.networks.decoder.memory_usage_flags.use_memory=true \
    slowrl.networks.decoder.memory_usage_flags.budget_trick=true \
    slowrl.networks.decoder.memory_usage_flags.remaining_budget_start=0.2 \
    slowrl.networks.decoder.memory_usage_flags.steps_trick=false \

# 1000 instances of size 200 (loaded)
python experiments/slowrl_validate.py \
    slowrl.environment.num_cities=200 \
    slowrl.memory.num_nodes=200 \
    slowrl.memory.memory_size=80 \
    slowrl.num_starting_points=-1 \
    slowrl.budget=1600 \
    slowrl.num_cmaes_states=2 \
    slowrl.validation_pop_size=16 \
    slowrl.rollout.policy.temperature=0.1 \
    slowrl.networks.decoder.memory_processing.mlp.num_layers=2 \
    slowrl.networks.decoder.memory_processing.mlp.hidden_size=8 \
    slowrl.problem_seed=0 \
    slowrl.problems.load_problem=True \
    slowrl.problems.load_path="data/validation/tsp200_test_small_seed1235.pkl" \
    slowrl.instances_batch_size=25 \
    slowrl.problems.num_problems=1000 \
    slowrl.networks.encoder.query_chunk_size=200 \
    slowrl.networks.decoder.query_chunk_size=200 \
    slowrl.networks.decoder.key_chunk_size=200 \
    slowrl.checkpointing.restore_path="data/memento/checkpoints/tsp" \
    slowrl.networks.decoder.memory_usage_flags.use_memory=true \
    slowrl.networks.decoder.memory_usage_flags.budget_trick=true \
    slowrl.networks.decoder.memory_usage_flags.remaining_budget_start=0.2 \
    slowrl.networks.decoder.memory_usage_flags.steps_trick=false \

# 1000 instances of size 100 (loaded)
python experiments/slowrl_validate.py \
    slowrl.environment.num_cities=100 \
    slowrl.memory.num_nodes=100 \
    slowrl.memory.memory_size=80 \
    slowrl.num_starting_points=-1 \
    slowrl.budget=1600 \
    slowrl.num_cmaes_states=3 \
    slowrl.validation_pop_size=16 \
    slowrl.rollout.policy.temperature=1 \
    slowrl.networks.decoder.memory_processing.mlp.num_layers=2 \
    slowrl.networks.decoder.memory_processing.mlp.hidden_size=8 \
    slowrl.problem_seed=0 \
    slowrl.problems.load_problem=True \
    slowrl.problems.load_path="data/validation/tsp100_test_seed1234.pkl" \
    slowrl.instances_batch_size=125 \
    slowrl.problems.num_problems=10000 \
    slowrl.networks.encoder.query_chunk_size=100 \
    slowrl.networks.decoder.query_chunk_size=100 \
    slowrl.networks.decoder.key_chunk_size=100 \
    slowrl.checkpointing.restore_path="data/memento/checkpoints/tsp" \
    slowrl.networks.decoder.memory_usage_flags.use_memory=true \
    slowrl.networks.decoder.memory_usage_flags.budget_trick=true \
    slowrl.networks.decoder.memory_usage_flags.remaining_budget_start=0.2 \
    slowrl.networks.decoder.memory_usage_flags.steps_trick=false \



}
