# EECS6322-project

## Dependencies
### python3
```python
pip install -r requirements.txt
```

### JAX
```python
pip install jax==0.2.8
pip install jaxlib==0.1.56+cuda100 -f https://storage.googleapis.com/jax-releases/jax_releases.html
```


### Running Unit Tests
```python
python unit_test/US1.02/test_vae_training.py
```

## Experiments Commands

## Toy Datasets(AbsFlow and AbsUnif Experiments)

Command for `checkerboard`:
```python
python experiments/toy/train_abs_unif.py --hidden_units [200,100] --dataset checkerboard --clim 0.05
```
Command for `corners`:
```python
python experiments/toy/train_abs_flow.py --hidden_units [200,100] --dataset corners --clim 0.1 --scale_fn softplus
```
Command for `eightgaussians`:
```python
python experiments/toy/train_abs_flow.py --hidden_units [200,100] --dataset eight_gaussians --clim 0.15 --scale_fn softplus
```
Command for `fourcircle`:
```python
python experiments/toy/train_abs_flow.py --hidden_units [200,100] --dataset fourcircle --clim 0.2 --scale_fn softplus
```

### Max Pooling Experiment
Command for `pool = none`
```python
python experiments/max_pooling/max_pooling_experiment.py --epochs 500 --batch_size 32 --optimizer adamax --lr 1e-4 --gamma 0.995 --eval_every 1 --check_every 10 --warmup 5000 --num_steps 12 --num_scales 2 --dequant flow --pooling none --dataset cifar10 --augmentation eta --name nonpool --model_dir ./experiments/max_pooling/checkpoints/
```
Command for `pool = max`
```python
python experiments/max_pooling/max_pooling_experiment.py --epochs 500 --batch_size 32 --optimizer adamax --lr 1e-4 --gamma 0.995 --eval_every 1 --check_every 10 --warmup 5000 --num_steps 12 --num_scales 2 --dequant flow --pooling max --dataset cifar10 --augmentation eta --name maxpool --model_dir ./experiments/max_pooling/checkpoints/
```

### Vincent's Experiment

### Stretch Goal First approach

### Stretch Goal Second approach
```python
python experiments/pro_nf_2/pro_nf.py --batch_size 32 --augmentation eta --dataset cifar10 --image_size 32
python experiments/pro_nf_2/pro_nf.py --batch_size 32 --augmentation eta --dataset cifar10 --image_size 16 --smallest
python experiments/pro_nf_2/pro_nf.py --batch_size 32 --augmentation eta --dataset cifar10 --image_size 32 16 --resume
```
