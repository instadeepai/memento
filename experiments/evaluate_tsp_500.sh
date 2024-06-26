#!/bin/bash
{
    #### LOW BUDGET ####
# default 50sp, 500 attempts = 25000
# used as for EAS, hence 50sp, budget 500
# COMPASS will use 10sp (and latent space 10), hence 250 attempts -> budget 2500

# COMPASS
python experiments/slowrl_validate.py \
    slowrl.environment.num_cities=500 \
    slowrl.num_starting_points=5 \
    slowrl.batch_size=8 \
    slowrl.num_cmaes_states=1 \
    slowrl.validation_pop_size=10 \
    slowrl.rollout.policy.temperature=0.5 \
    slowrl.problems.load_problem=true \
    slowrl.problems.num_problems=128 \
    slowrl.problems.load_path="data/validation/test-500-coords.pkl" \
    slowrl.budget=5000 \
    slowrl.checkpointing.restore_path="data/memento/checkpoints/tsp/" \
    slowrl.batch_size=8 \
    slowrl.networks.encoder.query_chunk_size=250 \
    slowrl.networks.decoder.query_chunk_size=500 \
    slowrl.networks.decoder.key_chunk_size=500

# EAS
python experiments/eas.py \
    eas.environment.num_cities=500 \
    eas_training=true \
    eas.num_starting_points=50 \
    eas.batch_size=8 \
    eas.problems.load_problem=true \
    eas.problems.num_problems=128 \
    eas.problems.load_path="data/validation/test-500-coords.pkl" \
    eas.budget=500 \
    eas.checkpointing.restore_path="data/memento/checkpoints/tsp/" \
    eas.policy.temperature=1 \
    eas.networks.encoder.query_chunk_size=250 \
    eas.networks.decoder.query_chunk_size=500 \
    eas.networks.decoder.key_chunk_size=500


# HIGH BUDGET
# default 100sp, 1000 attempts = 100000
# used as for EAS, hence 100sp, budget 1000
# COMPASS will use 20sp (and latent space 20), hence 500 attempts -> budget 5000

# COMPASS
python experiments/slowrl_validate.py \
    slowrl.environment.num_cities=500 \
    slowrl.num_starting_points=10 \
    slowrl.batch_size=8 \
    slowrl.num_cmaes_states=1 \
    slowrl.validation_pop_size=10 \
    slowrl.rollout.policy.temperature=0.5 \
    slowrl.problems.load_problem=true \
    slowrl.problems.num_problems=128 \
    slowrl.problems.load_path="data/validation/test-500-coords.pkl" \
    slowrl.budget=10000 \
    slowrl.checkpointing.restore_path="data/memento/checkpoints/tsp/" \
    slowrl.batch_size=8 \
    slowrl.networks.encoder.query_chunk_size=250 \
    slowrl.networks.decoder.query_chunk_size=500 \
    slowrl.networks.decoder.key_chunk_size=500

# EAS
python experiments/eas.py \
    eas.environment.num_cities=500 \
    eas_training=true \
    eas.num_starting_points=100 \
    eas.batch_size=8 \
    eas.problems.load_problem=true \
    eas.problems.num_problems=128 \
    eas.problems.load_path="data/validation/test-500-coords.pkl" \
    eas.budget=1000 \
    eas.checkpointing.restore_path="data/memento/checkpoints/tsp/" \
    eas.policy.temperature=1 \
    eas.networks.encoder.query_chunk_size=250 \
    eas.networks.decoder.query_chunk_size=500 \
    eas.networks.decoder.key_chunk_size=500

}