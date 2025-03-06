#!/bin/bash

python3.11 -m venv .env

source .env/bin/activate

pip install --upgrade pip

pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 xformers --index-url https://download.pytorch.org/whl/cu118 --no-cache-dir

pip install tensorflow[and-cuda] --no-cache-dir

pip install -r requirement.txt --no-cache-dir

