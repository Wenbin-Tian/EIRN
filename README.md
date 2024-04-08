# Enhanced Invertible Rescaling Network for Neural Video Delivery
This is the Pytorch implementation of paper: Enhanced Invertible Rescaling Network for Neural Video Delivery.
## Requirement
- Python 3.8(Recommend to use Anaconda)
- PyTorch >= 1.5
- NVIDIA GPU + CUDA
- Python packages: pip install numpy opencv-python lmdb pyyaml tb-nightly future

or you can use our environments(eirn_env.yml), it can be found EIRN/, then `conda env create -f eirn_env.yml`
## Dataset
The dataset(VSD4K) can be founded in [CaFM-Pytorch-ICCV2021](https://github.com/Neural-video-delivery/CaFM-Pytorch-ICCV2021)
Then download and organize
```
path/datasets/DIV2K
├── DIV2K_train_HR
└── DIV2K_train_LR_bicubic
    └── X2
        └─ 00001_x2.png
    └── X3
        └─ 00001_x3.png
    └── X4
        └─ 00001_x4.png
e.g.
home/lee/datasets/DIV2K
├── DIV2K_train_HR
└──DIV2K_train_LR_bicubic
    └── X2
    └── X3
    └── X4
```
