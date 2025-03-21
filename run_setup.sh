#!/usr/bin/env bash

export HUGGING_FACE_HUB_TOKEN="your_hugging_face_token"
export WANDB_API_KEY="your_wandb_token"

apt update
pip install --upgrade pip
pip install -r requirements.txt
pip uninstall transformer-engine -y