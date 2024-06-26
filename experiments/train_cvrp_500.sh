#!/bin/bash

{
    python experiments/train.py --config-name config_exp_cvrp \
    environment.num_nodes=500 \
    environment.norm_factor=100 \
    batch_size=8 \
    training_sample_size=32 \
    behavior_amplification=100 \
    optimizer.num_gradient_accumulation_steps=8 \
    num_starting_positions=20 \
    networks.encoder.query_chunk_size=501 \
    networks.decoder.query_chunk_size=500 \
    networks.decoder.query_chunk_size=500 \
    optimizer.encoder.lr=0.0001  \
    optimizer.decoder.lr=0.0001  \
    checkpointing.restore_path="data/memento/checkpoints/cvrp/" \
    slowrl.environment.num_nodes=500 \
    slowrl.environment.norm_factor=100 \
    slowrl.num_starting_points=40 \
    slowrl.budget=200 \
    slowrl.networks.encoder.query_chunk_size=501 \
    slowrl.networks.decoder.key_chunk_size=500 \
    slowrl.networks.decoder.query_chunk_size=500 \
    validation.environment.num_nodes=500 \
    validation.environment.norm_factor=100 \
    validation.num_starting_points=40 \
    validation.networks.encoder.query_chunk_size=501 \
    validation.networks.decoder.key_chunk_size=500 \
    validation.networks.decoder.query_chunk_size=500 
}