# iid
mpirun -np 11 python main_sync.py --algor fedavg --dataset fashionmnist --iid
mpirun -np 11 python main_sync.py --algor fedprox --dataset fashionmnist --iid
mpirun -np 11 python main_async.py --algor fedasync --dataset fashionmnist --iid
mpirun -np 11 python main_async.py --algor aflcs --dataset fashionmnist --iid
mpirun -np 11 python main_async.py --algor fedbuff --dataset fashionmnist --iid
# iid epochs(3)
mpirun -np 11 python main_sync.py --algor fedavg --dataset fashionmnist --epochs 3 --iid
mpirun -np 11 python main_sync.py --algor fedprox --dataset fashionmnist --epochs 3 --iid
mpirun -np 11 python main_async.py --algor fedasync --dataset fashionmnist --epochs 3 --iid
mpirun -np 11 python main_async.py --algor aflcs --dataset fashionmnist --epochs 3 --iid
mpirun -np 11 python main_async.py --algor fedbuff --dataset fashionmnist --epochs 3 --iid
# iid epochs(5)
mpirun -np 11 python main_sync.py --algor fedavg --dataset fashionmnist --epochs 5 --iid
mpirun -np 11 python main_sync.py --algor fedprox --dataset fashionmnist --epochs 5 --iid
mpirun -np 11 python main_async.py --algor fedasync --dataset fashionmnist --epochs 5 --iid
mpirun -np 11 python main_async.py --algor aflcs --dataset fashionmnist --epochs 5 --iid
mpirun -np 11 python main_async.py --algor fedbuff --dataset fashionmnist --epochs 5 --iid
# non-iid(0.1)
mpirun -np 11 python main_sync.py --algor fedavg --dataset fashionmnist --alpha 0.1
mpirun -np 11 python main_sync.py --algor fedprox --dataset fashionmnist --alpha 0.1
mpirun -np 11 python main_async.py --algor fedasync --dataset fashionmnist --alpha 0.1
mpirun -np 11 python main_async.py --algor aflcs --dataset fashionmnist --alpha 0.1
mpirun -np 11 python main_async.py --algor fedbuff --dataset fashionmnist --alpha 0.1
# non-iid(0.01)
mpirun -np 11 python main_sync.py --algor fedavg --dataset fashionmnist --alpha 0.01
mpirun -np 11 python main_sync.py --algor fedprox --dataset fashionmnist --alpha 0.01
mpirun -np 11 python main_async.py --algor fedasync --dataset fashionmnist --alpha 0.01
mpirun -np 11 python main_async.py --algor aflcs --dataset fashionmnist --alpha 0.01
mpirun -np 11 python main_async.py --algor fedbuff --dataset fashionmnist --alpha 0.01
# non-iid(0.5)
mpirun -np 11 python main_sync.py --algor fedavg --dataset fashionmnist --alpha 0.5
mpirun -np 11 python main_sync.py --algor fedprox --dataset fashionmnist --alpha 0.5
mpirun -np 11 python main_async.py --algor fedasync --dataset fashionmnist --alpha 0.5
mpirun -np 11 python main_async.py --algor aflcs --dataset fashionmnist --alpha 0.5
mpirun -np 11 python main_async.py --algor fedbuff --dataset fashionmnist --alpha 0.5
