#!/bin/bash

python3.12 -m venv .env

source .env/bin/activate

pip install --upgrade pip

pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128 --no-cache-dir

pip install tensorflow[and-cuda] --no-cache-dir

pip install -r requirement.txt --no-cache-dir

