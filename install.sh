#!/usr/bin/env bash
set -e
ENV_NAME=aidd_test

# create conda env
source "$(conda info --base)/etc/profile.d/conda.sh"

conda env create -n $ENV_NAME -f environment.yml
conda activate $ENV_NAME

# pip installation
pip install -r requirements.txt

echo "Done. 运行 conda activate $ENV_NAME 即可使用"
