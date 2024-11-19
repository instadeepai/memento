#!/bin/bash

{

python experiments/train.py --config-name config_exp_cvrp \
    environment.num_nodes=500 \
    environment.norm_factor=100 \
    memory.num_nodes=500 \
    memory.memory_size=40 \
    num_steps=6000000 \
    batch_size=8 \
    num_starting_positions=100 \
    budget=200 \
    optimizer.num_gradient_accumulation_steps=800 \
    loss.weight_offset=0.0000001 loss.weight_scale=1 \
    checkpointing.restore_path="data/memento/checkpoints/cvrp" \
    checkpointing.checkpoint_fname_load="pomo_500" \
    networks.encoder.query_chunk_size=501    \
    networks.decoder.query_chunk_size=500    \
    networks.decoder.key_chunk_size=500 \
    networks.decoder.memory_processing.mlp.num_layers=2 \
    networks.decoder.memory_processing.mlp.hidden_size=8 \
    slowrl.environment.num_nodes=500 \
    slowrl.environment.norm_factor=100 \
    slowrl.memory.num_nodes=500 \
    slowrl.memory.memory_size=80 \
    slowrl.num_starting_points=80 \
    slowrl.budget=200 \
    slowrl.networks.encoder.query_chunk_size=501 \
    slowrl.networks.decoder.query_chunk_size=501 \
    slowrl.networks.decoder.key_chunk_size=501 \
    slowrl.networks.decoder.memory_processing.mlp.num_layers=2 \
    slowrl.networks.decoder.memory_processing.mlp.hidden_size=8 \
    slowrl.problems.load_problem=false slowrl.problems.num_problems=32 \
    slowrl.instances_batch_size=4 \
    validation.environment.num_nodes=500 \
    validation.environment.norm_factor=100 \
    validation.networks.encoder.query_chunk_size=501   \
    validation.networks.decoder.query_chunk_size=501    \
    validation.networks.decoder.key_chunk_size=501 \
    validation.networks.decoder.memory_processing.mlp.num_layers=2 \
    validation.networks.decoder.memory_processing.mlp.hidden_size=8 

}