#!/bin/bash




pip install torch_geometric
pip install dive-into-graphs

pip install torch-scatter -f https://data.pyg.org/whl/torch-{torch.__version__}.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-{torch.__version__}.html
pip install torch-cluster -f https://data.pyg.org/whl/torch-{torch.__version__}.html
