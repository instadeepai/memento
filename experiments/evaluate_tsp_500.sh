#!/bin/bash

{

#### LOW BUDGET ####
# default 50sp, 500 attempts = 25000
# used as for MEMENTO, hence 50sp, budget 500
python experiments/slowrl_validate.py \
    slowrl.environment.num_cities=500 \
    slowrl.memory.num_nodes=500 \
    slowrl.num_starting_points=50 \
    slowrl.instances_batch_size=8 \
    slowrl.problems.load_problem=true \
    slowrl.problems.num_problems=128 \
    slowrl.problems.load_path="data/validation/test-500-coords.pkl" \
    slowrl.memory.memory_size=80 \
    slowrl.budget=500 \
    slowrl.networks.decoder.memory_processing.mlp.num_layers=2 \
    slowrl.networks.decoder.memory_processing.mlp.hidden_size=8 \
    slowrl.checkpointing.restore_path="data/memento/checkpoints/tsp" \
    slowrl.checkpointing.checkpoint_fname_load="memento_500" \
    slowrl.rollout.policy.temperature=0.8 \
    slowrl.networks.encoder.query_chunk_size=250 \
    slowrl.networks.decoder.query_chunk_size=500 \
    slowrl.networks.decoder.key_chunk_size=500

#### HIGH BUDGET ####
# default 100sp, 1000 attempts
# used as for MEMENTO, hence 100sp, budget 1000
python experiments/slowrl_validate.py \
    slowrl.environment.num_cities=500 \
    slowrl.memory.num_nodes=500 \
    slowrl.num_starting_points=100 \
    slowrl.instances_batch_size=8 \
    slowrl.problems.load_problem=true \
    slowrl.problems.num_problems=128 \
    slowrl.problems.load_path="data/validation/test-500-coords.pkl" \
    slowrl.memory.memory_size=80 \
    slowrl.budget=1000 \
    slowrl.networks.decoder.memory_processing.mlp.num_layers=2 \
    slowrl.networks.decoder.memory_processing.mlp.hidden_size=8 \
    slowrl.checkpointing.restore_path="data/memento/checkpoints/tsp" \
    slowrl.checkpointing.checkpoint_fname_load="memento_500" \
    slowrl.rollout.policy.temperature=0.8 \
    slowrl.networks.encoder.query_chunk_size=250 \
    slowrl.networks.decoder.query_chunk_size=500 \
    slowrl.networks.decoder.key_chunk_size=500

}
