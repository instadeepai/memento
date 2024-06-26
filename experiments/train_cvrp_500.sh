#!/bin/bash

{

python experiments/train.py --config-name config_exp_cvrp \
    environment.num_nodes=500 \
    environment.norm_factor=100 \
    batch_size=32 \
    num_starting_positions=200 \
    num_steps=6000000 \
    networks.encoder.query_chunk_size=501 \
    networks.decoder.query_chunk_size=500 \
    networks.decoder.key_chunk_size=500 \
    validation.environment.num_nodes=500 \
    validation.environment.norm_factor=100 \
    validation.networks.encoder.query_chunk_size=501 \
    validation.networks.decoder.query_chunk_size=500 \
    validation.networks.decoder.key_chunk_size=500 

}    