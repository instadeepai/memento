#!/bin/bash

{

python experiments/train.py environment.num_cities=100 \
    memory.num_nodes=100 \
    memory.memory_size=40 \
    num_steps=100000000 \
    training_sample_size=4 \
    batch_size=64 \
    pop_size=1 \
    num_starting_positions=40 \
    budget=200 \
    loss.sp_spec=false \
    optimizer.num_gradient_accumulation_steps=200 \
    optimizer.encoder.lr=0.0001 optimizer.decoder.lr=0.0001 \
    optimizer.memory.lr=0.004 \
    loss.weight_offset=0.0000001 loss.weight_scale=1 \
    checkpointing.restore_path="data/memento/checkpoints/tsp" \
    checkpointing.checkpoint_fname_load="compass_100" \
    networks.decoder.memory_processing.mlp.num_layers=2 \
    networks.decoder.memory_processing.mlp.hidden_size=8 \
    slowrl.environment.num_cities=100 \
    slowrl.memory.num_nodes=100 \
    slowrl.memory.memory_size=40 \
    slowrl.num_starting_points=20 \
    slowrl.budget=800 \
    slowrl.num_cmaes_states=1 \
    slowrl.rollout.policy.temperature=0.1 \
    slowrl.networks.decoder.memory_processing.mlp.num_layers=2 \
    slowrl.networks.decoder.memory_processing.mlp.hidden_size=8 \
    slowrl.problems.load_problem=false slowrl.problems.num_problems=32 \
    slowrl.instances_batch_size=4 \
    validation.networks.decoder.memory_processing.mlp.num_layers=2 \
    validation.networks.decoder.memory_processing.mlp.hidden_size=8 \
    checkpointing.restore_memory_path="data/memento/checkpoints/tsp" \
    checkpointing.memory_checkpoint_fname_load="memento_100" \
    networks.encoder.query_chunk_size=100 \
    networks.decoder.query_chunk_size=100 \
    networks.decoder.key_chunk_size=100 \
    validation.networks.encoder.query_chunk_size=100 \
    validation.networks.decoder.query_chunk_size=100 \
    validation.networks.decoder.key_chunk_size=100 \
    slowrl.networks.encoder.query_chunk_size=100 \
    slowrl.networks.decoder.query_chunk_size=100 \
    slowrl.networks.decoder.key_chunk_size=100 \

}