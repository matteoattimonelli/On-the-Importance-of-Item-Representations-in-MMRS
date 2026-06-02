#!/bin/sh
set -e

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

# Install PyTorch and Cython
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu118
pip install Cython==3.0.11

# Install PyG dependencies
pip install torch_scatter -f https://data.pyg.org/whl/torch-2.2.0+cu118.html
pip install torch_sparse -f https://data.pyg.org/whl/torch-2.2.0+cu118.html
pip install torch_cluster -f https://data.pyg.org/whl/torch-2.2.0+cu118.html
pip install torch_spline_conv -f https://data.pyg.org/whl/torch-2.2.0+cu118.html
pip install torch_geometric -f https://data.pyg.org/whl/torch-2.2.0+cu118.html
# Install the rest of the packages
pip install -r "$SCRIPT_DIR/venv.txt"
