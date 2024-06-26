#!/bin/bash

{
python experiments/train.py environment.num_cities=500 \
    batch_size=64 \
    optimizer.num_gradient_accumulation_steps=1 \
    num_starting_positions=200 \
    num_steps=6000000 \
    networks.encoder.query_chunk_size=250 \
    networks.decoder.query_chunk_size=500 \
    networks.decoder.key_chunk_size=500

}