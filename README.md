# Code

## Requirements

* Linux and Windows are supported, but we recommend Linux for performance and compatibility reasons.
* 1+ high-end NVIDIA GPU for sampling and 4+ GPUs for training. We have done all testing and development using NVIDIA L40S GPUs.
* 64-bit Python 3.8 and PyTorch 1.12.1 (or later). See https://pytorch.org for PyTorch install instructions.

## Data Preparation
You may leverage the data already downloaded in the local folder. Alternatively, you may prepare the data following the steps provided below:

### Copy the preprocessed data
Download the CIFAR10 data:
```sh
wget -P downloads/cifar10/ https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz

python dataset_tool.py --source=downloads/cifar10/cifar-10-python.tar.gz \
    --dest=datasets/cifar10-32x32.zip
```

## Training
training  on CIFAR10:

```bash
python -m torch.distributed.run --standalone --nproc_per_node=4  train.py --outdir=train-runs/ --data=datasets/cifar10-32x32.zip --cond=False --arch=ddpmpp --batch=288 --precond=diff --lr=2e-4 --Shift=0.60 --Scale=0.39 --sigmoid_start=10 --sigmoid_end=-13 --sigmoid_power=1 --lossType='PHK' --eta=10000
```

## FID Evaluation

Note that the numerical value of FID varies across different random seeds and is highly sensitive to the number of images. By default, fid.py will always use 50,000 generated images; providing fewer images will result in an error, whereas providing more will use a random subset. To reduce the effect of random variation, we recommend repeating the calculation multiple times with different seeds, e.g., --seeds=0-49999, --seeds=50000-99999, and --seeds=100000-149999. 

```bash
#Generate 50000 images
python -m torch.distributed.run --standalone --nproc_per_node=4 generate.py --steps=200 --outdir=plots/images --network=plots/00001-cifar10-32x32-uncond-ddpmpp-diff-gpus4-batch288-fp32/network-snapshot-200000.pkl --seeds=0-49999
#Calculate the FID of x0_hat
python -m torch.distributed.run --standalone --nproc_per_node=4 fid.py calc --images=plots/images --ref=$fid_file
```