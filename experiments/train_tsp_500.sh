#!/bin/bash

{
    python experiments/train.py environment.num_cities=500 \
    batch_size=8 \
    training_sample_size=64 \
    behavior_amplification=100 \
    optimizer.num_gradient_accumulation_steps=8 \
    num_starting_positions=30 \
    networks.encoder.query_chunk_size=250    \
    checkpointing.restore_path="data/memento/checkpoints/tsp/" \
    slowrl.environment.num_cities=500 \
    slowrl.num_starting_points=40 \
    slowrl.budget=200 \
    slowrl.networks.encoder.query_chunk_size=250 \
    validation.environment.num_cities=500 \
    validation.networks.encoder.query_chunk_size=250 
}