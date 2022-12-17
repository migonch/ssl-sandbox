# ssl-sandbox
Sandbox for comparison self-supervised learning methods

## Installation
Make sure that you have installed [torch](https://pytorch.org/) compatible with your CUDA version.

Then use
```
git clone https://github.com/mishagoncharov/ssl-sandbox.git && cd ssl-sandbox && pip install -e .
```

## Experiments
To train and validate a supervised classifier
- on MNIST dataset, run
```
python scripts/image2vec.py --dataset mnist --mnist_dir <where/to/save/mnist/data> --logs_dir <where/to/save/logs> --supervised
```
- on CIFAR10 dataset, run
```
python scripts/image2vec.py --dataset cifar10 --cifar10_dir <where/to/save/cifar10/data> --logs_dir <where/to/save/logs> --supervised
```
Note that **MNIST and CIFAR10 datasets are downloaded automatically**.

To train and validate a vanïla autoencoder, run
```
python scripts/image2vec.py --dataset cifar10 --cifar10_dir <cifar10_dir> --logs_dir <logs_dir> --ae
```

To train and validate a variational autoencoder (VAE), run
```
python scripts/image2vec.py --dataset cifar10 --cifar10_dir <cifar10_dir> --logs_dir <logs_dir> --vae
```

To change a dimensionality of the VAE's latent space, run
```
python scripts/image2vec.py --dataset cifar10 --cifar10_dir <cifar10_dir> --logs_dir <logs_dir> --vae --vae_latent_dim 32
```

To train and validate a SimCLR model, run
```
python scripts/image2vec.py --dataset cifar10 --cifar10_dir <cifar10_dir> --logs_dir <logs_dir> --simclr
```

To train and validate a model which combines VAE and SimCLR, run 
```
python scripts/image2vec.py --dataset cifar10 --cifar10_dir <cifar10_dir> --logs_dir <logs_dir> --vae --simclr
```
