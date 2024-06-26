#!/bin/bash
{
  #### LOW BUDGET ####
# default 50sp, 500 attempts = 25000
# used as for EAS, hence 50sp, budget 500
# COMPASS will use 10sp (and latent space 10), hence 250 attempts -> budget 2500


# COMPASS
python experiments/slowrl_validate.py --config-name config_exp_cvrp \
    slowrl.environment.num_nodes=500 \
    slowrl.environment.norm_factor=100 \
    slowrl.num_starting_points=20 \
    slowrl.validation_pop_size=10 \
    slowrl.num_cmaes_states=1 \
    slowrl.budget=1250 \
    slowrl.problems.load_problem=True \
    slowrl.problems.load_path="data/validation/vrp500_test_lkh.pkl" \
    slowrl.batch_size=8 \
    slowrl.problems.num_problems=128 \
    slowrl.rollout.policy.temperature=0.3 \
    slowrl.networks.encoder.query_chunk_size=501 \
    slowrl.networks.decoder.query_chunk_size=501 \
    slowrl.networks.decoder.key_chunk_size=501 \
    slowrl.checkpointing.restore_path="data/memento/checkpoints/cvrp/"


# run EAS 
python experiments/eas.py --config-name config_exp_cvrp \
    eas_training=rue \
    eas.environment.num_nodes=500 \
    eas.environment.norm_factor=100 \
    eas.num_starting_points=50 \
    eas.budget=500 \
    eas.problems.load_problem=True \
    eas.problems.load_path="data/validation/vrp500_test_lkh.pkl" \
    eas.batch_size=8 \
    eas.problems.num_problems=128 \
    eas.policy.temperature=1 \
    eas.networks.encoder.query_chunk_size=501 \
    eas.networks.decoder.query_chunk_size=501 \
    eas.networks.decoder.key_chunk_size=501 \
    eas.checkpointing.restore_path="data/memento/checkpoints/cvrp/"


#### HIGH BUDGET ####
# default 100sp, 1000 attempts
# used as for EAS, hence 100sp, budget 1000
# COMPASS will use 40sp (and latent space 10), hence 250 attempts -> budget 2500

# COMPASS
python experiments/slowrl_validate.py --config-name config_exp_cvrp \
    slowrl.environment.num_nodes=500 \
    slowrl.environment.norm_factor=100 \
    slowrl.num_starting_points=40 \
    slowrl.validation_pop_size=10 \
    slowrl.num_cmaes_states=1 \
    slowrl.budget=2500 \
    slowrl.problems.load_problem=True \
    slowrl.problems.load_path="data/validation/vrp500_test_lkh.pkl" \
    slowrl.batch_size=8 \
    slowrl.problems.num_problems=128 \
    slowrl.rollout.policy.temperature=0.3 \
    slowrl.networks.encoder.query_chunk_size=501 \
    slowrl.networks.decoder.query_chunk_size=501 \
    slowrl.networks.decoder.key_chunk_size=501 \
    slowrl.checkpointing.restore_path="data/memento/checkpoints/cvrp/"


# run EAS 
python experiments/eas.py --config-name config_exp_cvrp \
    eas_training=True \
    eas.environment.num_nodes=500 \
    eas.environment.norm_factor=100 \
    eas.num_starting_points=100 \
    eas.budget=1000 \
    eas.problems.load_problem=True \
    eas.problems.load_path="data/validation/vrp500_test_lkh.pkl" \
    eas.batch_size=8 \
    eas.problems.num_problems=128 \
    eas.policy.temperature=1 \
    eas.networks.encoder.query_chunk_size=501 \
    eas.networks.decoder.query_chunk_size=501 \
    eas.networks.decoder.key_chunk_size=501 \
    eas.checkpointing.restore_path="data/memento/checkpoints/cvrp/"
  
}