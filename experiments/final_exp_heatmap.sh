#!/bin/bash

{
# COMPASS ON TSP500 
for sp in 5 10 20 30 40 50 60 70 80 90 100 110 120 130 140
do
    python experiments/slowrl_validate.py \
        slowrl.environment.num_cities=500 \
        slowrl.num_starting_points=$sp \
        slowrl.batch_size=8 \
        slowrl.num_cmaes_states=1 \
        slowrl.validation_pop_size=10 \
        slowrl.rollout.policy.temperature=0.5 \
        slowrl.problems.load_problem=true \
        slowrl.problems.num_problems=64 \
        slowrl.problems.load_path="data/validation/tsp500_test_small_seed1235.pkl" \
        slowrl.budget=10000 \
        slowrl.checkpointing.restore_path="data/memento/checkpoints/tsp/" \
        slowrl.batch_size=8 \
        slowrl.networks.encoder.query_chunk_size=250 \
        slowrl.networks.decoder.query_chunk_size=500 \
        slowrl.networks.decoder.key_chunk_size=500

done

# COMPASS ON CVRP500 
for sp in 5 10 20 30 40 50 60 70 80 90 100 110 120 130 140
do
        python experiments/slowrl_validate.py --config-name config_exp_cvrp \
            slowrl.environment.num_nodes=500 \
            slowrl.environment.norm_factor=100 \
            slowrl.num_starting_points=$sp \
            slowrl.ref_pop_size=10 \
            slowrl.num_cmaes_states=1 \
            slowrl.budget=10000 \
            slowrl.problems.load_problem=True \
            slowrl.problems.load_path="data/validation/vrp500_test_small_seed1235.pkl" \
            slowrl.batch_size=8 \
            slowrl.problems.num_problems=64 \
            slowrl.rollout.policy.temperature=0.1 \
            slowrl.networks.encoder.query_chunk_size=501 \
            slowrl.networks.decoder.query_chunk_size=500 \
            slowrl.networks.decoder.key_chunk_size=500 \
            slowrl.checkpointing.restore_path="data/memento/checkpoints/cvrp/" \
    done
done 

# EAS ON TSP500
for sp in 5 10 20 30 40 50 60 70 80 90 100 110 120 140
do
    python experiments/eas.py \
        eas_training=True \
        eas.environment.num_cities=500 \
        eas.num_starting_points=$sp \
        eas.batch_size=8 \
        eas.problems.load_problem=true \
        eas.problems.num_problems=64 \
        eas.problems.load_path="data/validation/tsp500_test_small_seed1235.pkl" \
        eas.budget=1000 \
        eas.checkpointing.restore_path="data/memento/checkpoints/tsp/" \
        eas.policy.temperature=1 \
        eas.networks.encoder.query_chunk_size=250 \
        eas.networks.decoder.query_chunk_size=500 \
        eas.networks.decoder.key_chunk_size=500

done


# EAS ON CVRP500
for sp in 5 10 20 30 40 50 60 70 80 90 100 110 120 130 140
do
    python experiments/eas.py --config-name config_exp_cvrp \
        eas_training=True \
        eas.environment.num_nodes=500 \
        eas.environment.norm_factor=100 \
        eas.num_starting_points=$sp \
        eas.budget=1000 \
        eas.problems.load_problem=True \
        eas.problems.load_path="data/validation/vrp500_test_small_seed1235.pkl" \
        eas.batch_size=8 \
        eas.problems.num_problems=64 \
        eas.policy.temperature=1 \
        eas.networks.encoder.query_chunk_size=501 \
        eas.networks.decoder.query_chunk_size=500 \
        eas.networks.decoder.key_chunk_size=500 \
        eas.checkpointing.restore_path="data/memento/checkpoints/cvrp/" \
done

}