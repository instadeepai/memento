#!/bin/bash

{
python experiments/train.py environment.num_cities=500 \
    memory.num_nodes=500 \
    memory.memory_size=80 \
    num_steps=5400000 \
    batch_size=32 \
    num_starting_positions=30 \
    budget=200 \
    optimizer.num_gradient_accumulation_steps=400 \
    rollout.policy.temperature=1 \
    loss.weight_offset=0.0000001 loss.weight_scale=1 \
    checkpointing.restore_path="data/memento/checkpoints/tsp" \
    checkpointing.checkpoint_fname_load="pomo_500" \
    networks.decoder.memory_processing.mlp.num_layers=2 \
    networks.decoder.memory_processing.mlp.hidden_size=8 \
    slowrl.environment.num_cities=500 \
    slowrl.memory.num_nodes=500 \
    slowrl.memory.memory_size=80 
    
}