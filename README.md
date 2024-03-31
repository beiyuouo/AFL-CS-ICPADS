# AFL-CS

Code implementation of the paper "AFL-CS: Asynchronous Federated Learning with Cosine Similarity-based Penalty Term and Aggregation" published in IEEE ICPADS 2023.

## How to run

Install the required packages:

```bash
conda create -n aflcs python=3.8
conda activate aflcs
conda install mpi4py
conda install pytorch==1.12.1 torchvision==0.13.1 -c pytorch
pip install -r requirements.txt
python utils.py # download the datasets 
```

Example:

```bash
bash scripts/run_mnist.sh
bash scripts/run_fashionmnist.sh
```