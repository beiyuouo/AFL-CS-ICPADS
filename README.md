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

## Citation

If you use this code for your research, please cite our paper:

```
@inproceedings{yan2023afl,
  title={AFL-CS: Asynchronous Federated Learning with Cosine Similarity-based Penalty Term and Aggregation},
  author={Yan, Bingjie and Jiang, Xinlong and Chen, Yiqiang and Gao, Chenlong and Liu, Xuequn},
  booktitle={2023 IEEE 29th International Conference on Parallel and Distributed Systems (ICPADS)},
  pages={46--53},
  year={2023},
  organization={IEEE}
}
```