source ~/env/bin/activate
for seed in {10..15}
do
  python3 experiment_mnist_rl.py --env mnist  --algo $1 --device $2 --n_gen 1000 --seed $seed
  python3 experiment_mnist_rl.py --env fmnist --algo $1 --device $2 --n_gen 1000 --seed $seed
done
deactivate
