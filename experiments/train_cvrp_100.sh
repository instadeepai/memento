#!/bin/bash

{
python experiments/train.py --config-name config_exp_cvrp \
    environment.num_nodes=100 \
    environment.norm_factor=50 \
    memory.num_nodes=100 \
    memory.memory_size=40 \
    num_steps=4550000 \
    batch_size=64 \
    num_starting_positions=100 \
    budget=200 \
    optimizer.num_gradient_accumulation_steps=200 \
    loss.weight_offset=0.0000001 loss.weight_scale=1 \
    networks.decoder.memory_processing.mlp.num_layers=2 \
    networks.decoder.memory_processing.mlp.hidden_size=8 \
    slowrl.environment.num_nodes=100 \
    slowrl.environment.norm_factor=50 \
    slowrl.memory.num_nodes=100 \
    slowrl.memory.memory_size=40 \
    slowrl.num_starting_points=40 \
    slowrl.budget=200 \
    slowrl.networks.decoder.memory_processing.mlp.num_layers=2 \
    slowrl.networks.decoder.memory_processing.mlp.hidden_size=8 \
    slowrl.problems.load_problem=false \
    slowrl.problems.num_problems=512 \
    slowrl.instances_batch_size=64 \
    validation.networks.decoder.memory_processing.mlp.num_layers=2 \
    validation.networks.decoder.memory_processing.mlp.hidden_size=8 \
}