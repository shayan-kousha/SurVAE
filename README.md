# EECS6322-project

This project is aimed to reproduce the major results from two papers, [SurVAE Flows: Surjections to Bridge the Gap between VAEs and Flows](https://arxiv.org/abs/2007.02731), and  `Normalizing Flows with Multi-Scale Autoregressive Priors (Mahajan et al., 2020)` as well as a stretch goal of implementing the idea of ProNF. The reproduction is made in JAX library.

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


## Experiments Commands

### Toy Datasets(AbsFlow Experiment)

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

### MSAR-SCF Experiment
```python
python experiments/msar_scf/train_msar_scf.py --ckptdir "experiments/msar_scf/ckpt_sigmoid" --activation "sigmoid" --resume True --num_epochs 3000
```

### Stretch Goal First approach
```python
## 16x16 => 32x32
python experiments/pro_nf/train_pronf.py --ckptdir "experiments/pro_nf/ckpt_32" --resume True --warmup 50000  --ms
## 8x8 => 16x16
python experiments/pro_nf/train_pronf.py --ckptdir "experiments/pro_nf/ckpt_16" --resume True --warmup 50000 --input_res 16 --num_layers 2 --ms --learning_rate 1e-4 
## 4x4 => 8x8
python experiments/pro_nf/train_pronf.py --ckptdir "experiments/pro_nf/ckpt_8" --resume True --warmup 50000 --input_res 8 --num_layers 2 --ms --learning_rate 1e-4
## 4x4 unconditional
python experiments/pro_nf/train_pronf.py --ckptdir "experiments/pro_nf/ckpt_4" --resume True --warmup 50000 --input_res 4 
## chain-up
python experiments/pro_nf/merge.py --ckptdir "experiments/pro_nf" --resume True
```

### Stretch Goal Second approach
```python
python experiments/pro_nf_2/pro_nf.py --batch_size 32 --augmentation eta --dataset cifar10 --image_size 32
python experiments/pro_nf_2/pro_nf.py --batch_size 32 --augmentation eta --dataset cifar10 --image_size 16 --smallest
python experiments/pro_nf_2/pro_nf.py --batch_size 32 --augmentation eta --dataset cifar10 --image_size 32 16 --resume
```
