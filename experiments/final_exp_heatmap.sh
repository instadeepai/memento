#!/bin/bash

{

#### TSP500 ######
for i in 5 10 20 30 40 50 60 70 80 90 100 110 120
do

    python experiments/slowrl_validate.py \
        slowrl.environment.num_cities=500 \
        slowrl.memory.num_nodes=500 \
        slowrl.num_starting_points=$i \
        slowrl.instances_batch_size=8 \
        slowrl.problems.load_problem=true \
        slowrl.problems.num_problems=64 \
        slowrl.problems.load_path="data/validation/tsp500_test_small_seed1235.pkl" \
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

done

#### CVRPP500 ######
for i in 5 10 20 30 40 50 60 70 80 90 100 110 120
do
    python experiments/slowrl_validate.py --config-name config_exp_cvrp \
        slowrl.environment.num_nodes=500 \
        slowrl.environment.norm_factor=100 \
        slowrl.num_starting_points=$i \
        slowrl.problems.load_problem=True \
        slowrl.problems.load_path="data/validation/vrp500_test_small_seed1235.pkl" \
        slowrl.instances_batch_size=8 \
        slowrl.problems.num_problems=64 \
        slowrl.memory.num_nodes=500 \
        slowrl.memory.memory_size=40 \
        slowrl.budget=1000 \
        slowrl.networks.decoder.memory_processing.mlp.num_layers=2 \
        slowrl.networks.decoder.memory_processing.mlp.hidden_size=8 \
        slowrl.networks.encoder.query_chunk_size=501 \
        slowrl.networks.decoder.query_chunk_size=500 \
        slowrl.networks.decoder.key_chunk_size=500 \
        slowrl.checkpointing.restore_path="data/memento/checkpoints/cvrp" \
        slowrl.checkpointing.checkpoint_fname_load="memento_500" \
        slowrl.rollout.policy.temperature=0.3 

}

