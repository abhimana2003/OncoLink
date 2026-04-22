#!/usr/bin/env bash

echo "Starting Nutrition AI Project"

echo "Installing dependencies"
pip install --upgrade pip
pip install -r requirements.txt

python3 processing.py
python3 model.py
python3 agent.py

