#!/usr/bin/env bash
# Exit immediately if a command exits with a non-zero status
set -o errexit

echo "==> Upgrading pip..."
pip install --upgrade pip

echo "==> Installing CPU-only PyTorch (lightweight, optimized for cloud deployment)..."
pip install torch --index-url https://download.pytorch.org/whl/cpu

echo "==> Installing project dependencies from requirements.txt..."
pip install -r requirements.txt

echo "==> Downloading spaCy English model (en_core_web_sm)..."
python -m spacy download en_core_web_sm

echo "==> Build completed successfully!"
