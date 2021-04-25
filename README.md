# EECS6322-project

## Dependencies
python3 </br>
`pip install -r requirements.txt`

### JAX
`pip install jax==0.2.8` </br>
`pip install jaxlib==0.1.56+cuda100 -f https://storage.googleapis.com/jax-releases/jax_releases.html`

### Running Unit Test
`python unit_test/US1.02/test_vae_training.py `

## Run Experiments

### Ali's Experiment

### Max Pooling Experiment
`python experiments/max_pooling/max_pooling_experiment.py --epochs 500 --batch_size 32 --optimizer adamax --lr 1e-4 --gamma 0.995 --eval_every 1 --check_every 10 --warmup 5000 --num_steps 12 --num_scales 2 --dequant flow --pooling none --dataset cifar10 --augmentation eta --name nonpool --model_dir ./experiments/max_pooling/checkpoints/`</br>
`python experiments/max_pooling/max_pooling_experiment.py --epochs 500 --batch_size 32 --optimizer adamax --lr 1e-4 --gamma 0.995 --eval_every 1 --check_every 10 --warmup 5000 --num_steps 12 --num_scales 2 --dequant flow --pooling max --dataset cifar10 --augmentation eta --name maxpool --model_dir ./experiments/max_pooling/checkpoints/`

### Vincent's Experiment

### Stretch Goal First approach

### Stretch Goal Second approach
`python experiments/pro_nf_2/pro_nf.py --batch_size 32 --augmentation eta --dataset cifar10 --image_size 32`</br>
`python experiments/pro_nf_2/pro_nf.py --batch_size 32 --augmentation eta --dataset cifar10 --image_size 16 --smallest`</br>
`python experiments/pro_nf_2/pro_nf.py --batch_size 32 --augmentation eta --dataset cifar10 --image_size 32 16 --resume`</br>
