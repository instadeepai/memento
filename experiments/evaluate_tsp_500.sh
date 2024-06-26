#!/bin/bash

{
##### LOW BUDGET ####
# default 50sp, 500 attempts = 25000
python experiments/slowrl_validate.py \
    slowrl.environment.num_cities=500 \
    slowrl.num_starting_points=50 \
    slowrl.batch_size=8 \
    slowrl.problems.load_problem=true \
    slowrl.problems.num_problems=128 \
    slowrl.problems.load_path="data/validation/test-500-coords.pkl" \
    slowrl.budget=500 \
    slowrl.checkpointing.restore_path="data/memento/checkpoints/ysp/" \
    slowrl.rollout.policy.temperature=1 \
    slowrl.networks.encoder.query_chunk_size=250 \
    slowrl.networks.decoder.query_chunk_size=500 \
    slowrl.networks.decoder.key_chunk_size=500 \


#### HIGH BUDGET ####
# default 100sp, 1000 attempts
python experiments/slowrl_validate.py \
    slowrl.environment.num_cities=500 \
    slowrl.num_starting_points=100 \
    slowrl.batch_size=8 \
    slowrl.problems.load_problem=true \
    slowrl.problems.num_problems=128 \
    slowrl.problems.load_path="data/validation/test-500-coords.pkl" \
    slowrl.budget=1000 \
    slowrl.checkpointing.restore_path="data/memento/checkpoints/tsp/" \
    slowrl.rollout.policy.temperature=1 \
    slowrl.networks.encoder.query_chunk_size=250 \
    slowrl.networks.decoder.query_chunk_size=500 \
    slowrl.networks.decoder.key_chunk_size=500 \

}