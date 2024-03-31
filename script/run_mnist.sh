# iid
mpirun -np 11 python main_sync.py --algor fedavg --dataset mnist --iid
mpirun -np 11 python main_sync.py --algor fedprox --dataset mnist --iid
mpirun -np 11 python main_async.py --algor fedasync --dataset mnist --iid
mpirun -np 11 python main_async.py --algor aflcs --dataset mnist --iid
mpirun -np 11 python main_async.py --algor fedbuff --dataset mnist --iid
# iid epochs(3)
mpirun -np 11 python main_sync.py --algor fedavg --dataset mnist --epochs 3 --iid
mpirun -np 11 python main_sync.py --algor fedprox --dataset mnist --epochs 3 --iid
mpirun -np 11 python main_async.py --algor fedasync --dataset mnist --epochs 3 --iid
mpirun -np 11 python main_async.py --algor aflcs --dataset mnist --epochs 3 --iid
mpirun -np 11 python main_async.py --algor fedbuff --dataset mnist --epochs 3 --iid
# iid epochs(5)
mpirun -np 11 python main_sync.py --algor fedavg --dataset mnist --epochs 5 --iid
mpirun -np 11 python main_sync.py --algor fedprox --dataset mnist --epochs 5 --iid
mpirun -np 11 python main_async.py --algor fedasync --dataset mnist --epochs 5 --iid
mpirun -np 11 python main_async.py --algor aflcs --dataset mnist --epochs 5 --iid
mpirun -np 11 python main_async.py --algor fedbuff --dataset mnist --epochs 5 --iid
# non-iid(0.1)
mpirun -np 11 python main_sync.py --algor fedavg --dataset mnist --alpha 0.1
mpirun -np 11 python main_sync.py --algor fedprox --dataset mnist --alpha 0.1
mpirun -np 11 python main_async.py --algor fedasync --dataset mnist --alpha 0.1
mpirun -np 11 python main_async.py --algor aflcs --dataset mnist --alpha 0.1
mpirun -np 11 python main_async.py --algor fedbuff --dataset mnist --alpha 0.1
# non-iid(0.01)
mpirun -np 11 python main_sync.py --algor fedavg --dataset mnist --alpha 0.01
mpirun -np 11 python main_sync.py --algor fedprox --dataset mnist --alpha 0.01
mpirun -np 11 python main_async.py --algor fedasync --dataset mnist --alpha 0.01
mpirun -np 11 python main_async.py --algor aflcs --dataset mnist --alpha 0.01
mpirun -np 11 python main_async.py --algor fedbuff --dataset mnist --alpha 0.01
# non-iid(0.5)
mpirun -np 11 python main_sync.py --algor fedavg --dataset mnist --alpha 0.5
mpirun -np 11 python main_sync.py --algor fedprox --dataset mnist --alpha 0.5
mpirun -np 11 python main_async.py --algor fedasync --dataset mnist --alpha 0.5
mpirun -np 11 python main_async.py --algor aflcs --dataset mnist --alpha 0.5
mpirun -np 11 python main_async.py --algor fedbuff --dataset mnist --alpha 0.5
