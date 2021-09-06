source ~/env/bin/activate
for seed in {0..5}
do
  echo $seed
  python3 experiment_mnist_rl.py --env cartpole  --algo $1 --device cpu --n_gen 100 --seed $seed
  python3 experiment_mnist_rl.py --env mountain --algo $1 --device cpu --n_gen 100 --seed $seed
  python3 experiment_mnist_rl.py --env pendulum --algo $1 --device cpu --n_gen 100 --seed $seed
  python3 experiment_mnist_rl.py --env acrobot --algo $1 --device cpu --n_gen 100 --seed $seed
done
deactivate
