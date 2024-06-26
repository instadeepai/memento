#!/bin/bash

{
python experiments/train.py --config-name config_exp_cvrp \
    environment.num_nodes=100 \
    environment.norm_factor=50 \
    memory.memory_size=40 \
    num_steps=5800000 \
    training_sample_size=8 \
    batch_size=32 \
    num_starting_positions=20 \
    budget=200 \
    loss.sp_spec=false \
    optimizer.num_gradient_accumulation_steps=100 \
    optimizer.encoder.lr=0.0001 optimizer.decoder.lr=0.0001 \
    optimizer.memory.lr=0.004 \
    loss.weight_offset=0.0000001 loss.weight_scale=1 \
    checkpointing.restore_path="data/memento/checkpoints/cvrp" \
    checkpointing.checkpoint_fname_load="compass_100" \
    slowrl.environment.num_nodes=100 \
    slowrl.environment.norm_factor=50 \
    slowrl.memory.num_nodes=100 \
    slowrl.memory.memory_size=40 \
    slowrl.num_starting_points=20 \
    slowrl.budget=800 \
    slowrl.num_cmaes_states=1 \
    slowrl.rollout.policy.temperature=0.1 \
    slowrl.problems.load_problem=false slowrl.problems.num_problems=32 \
    slowrl.instances_batch_size=4 \
    validation.environment.num_nodes=100 \
    validation.environment.norm_factor=50 \
    checkpointing.restore_memory_path="data/memento/checkpoints/cvrp" \
    checkpointing.memory_checkpoint_fname_load="memento_100" \
    networks.decoder.memory_processing.mlp.num_layers=2 \
    networks.decoder.memory_processing.mlp.hidden_size=8 \
    validation.networks.decoder.memory_processing.mlp.num_layers=2 \
    validation.networks.decoder.memory_processing.mlp.hidden_size=8 \
    slowrl.networks.decoder.memory_processing.mlp.num_layers=2 \
    slowrl.networks.decoder.memory_processing.mlp.hidden_size=8 
}