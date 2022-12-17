# ssl-sandbox
Sandbox for comparison self-supervised learning methods

## Installation
Make sure that you have installed [torch](https://pytorch.org/) compatible with your CUDA version.

Then use
```
git clone https://github.com/mishagoncharov/ssl-sandbox.git && cd ssl-sandbox && pip install -e .
```

For now, to run experiments you also need to
1) Sign up at https://wandb.ai/site
2) Run `wandb login` in a command line and paste a key from https://wandb.ai/authorize.

## Experiments
To train and validate a supervised classifier on CIFAR10 dataset (it will be downloaded automatically), run
```
python scripts/cifar10/image2vec.py --cifar_dir <where/to/save/cifar/data> --logs_dir <where/to/save/logs> --name <wandb_run_name> --supervised 
```

To train and validate a vanïla autoencoder, run
```
python scripts/cifar10/image2vec.py --cifar_dir <where/to/save/cifar/data> --logs_dir <where/to/save/logs> --name <wandb_run_name> --ae
```

To train and validate a variational autoencoder (VAE), run
```
python scripts/cifar10/image2vec.py --cifar_dir <where/to/save/cifar/data> --logs_dir <where/to/save/logs> --name <wandb_run_name> --vae
```

To change a dimensionality of the VAE's latent space, run
```
python scripts/cifar10/image2vec.py --cifar_dir <where/to/save/cifar/data> --logs_dir <where/to/save/logs> --name <wandb_run_name> --vae --vae_latent_dim 32
```

To train and validate a SimCLR model, run
```
python scripts/cifar10/image2vec.py --cifar_dir <where/to/save/cifar/data> --logs_dir <where/to/save/logs> --name <wandb_run_name> --simclr
```

To train and validate a model which combines VAE and SimCLR, run 
```
python scripts/cifar10/image2vec.py --cifar_dir <where/to/save/cifar/data> --logs_dir <where/to/save/logs> --name <wandb_run_name> --vae --simclr
```
