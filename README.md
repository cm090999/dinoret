This repository is based on the implementation from RETFound: [Article](https://www.nature.com/articles/s41586-023-06555-x), [Github](https://github.com/rmaphoh/RETFound_MAE.git)

## BE-DINORET: Block Expanded DINOv2 for Retinal Imaging
This is the official repository for [Block Expanded DINORET: Adapting Natural Domain Foundation Models for Retinal Imaging Without Catastrophic Forgetting](https://arxiv.org/abs/2409.17332).

### Self-Supervised pretraining on retinal images
For running the pretraining for DINOv2 and the block expanded version, please refer to the official [DINOv2 repository](https://github.com/facebookresearch/dinov2.git) as well as the [official implementation of the block expansion](https://github.com/TencentARC/LLaMA-Pro.git).

### Key features

- Finetuning scripts for BE-DINOv2 and DINOv2 for retinal images
- RETFound finetuning and inference scripts

### Install environment

Our experiments were done locally on an Ubuntu 22.04 LTS computer.

1. Create environment:

```
bash scripts/create_env.sh
```

2. Activate environment:
```
source .env/bin/activate
```


3. Run distributed experiment

There is a dataloader that support the following dataset in their respective original structure after download: [aptos](https://www.kaggle.com/competitions/aptos2019-blindness-detection) [eyepacs](https://www.kaggle.com/competitions/diabetic-retinopathy-detection/data), [idrid](https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid), [messidor](https://www.adcis.net/en/third-party/messidor2/), [drtid](https://github.com/fdu-vts/drtid), [papila](https://figshare.com/articles/dataset/PAPILA/14798004/1?file=28454352)

The flag ```task``` defines the output folder for the results. Ensure that this folder exists before launching a training run.

To use BE DINORET, add the flag ```--block_expansion_positions "1 2 3"``` along with the path to the downloaded weights ```--pretrained_checkpoint /path/to/BEDINOERET```. ```"1 2 3"``` adds transformer blocks after the 1st, 2nd and 3rd original transformer block. As the weights are loaded from the pretrained BE DINORET, the positions do not matter, only 3 blocks need to be added.  

```
 torchrun main_finetune.py --data_path /path/to/dataset --task results/
```

### Weights

We provide the weights of DINORET and BE DINORET on a google drive share:

- DINORET (based on DINOv2 base):

https://drive.google.com/uc?export=download&id=1Dx-sMTjWRb9wgN5lLXrpYgNRl7yEG3Fr
- BE DINORET (DINOv2 base with 3 expanded blocks):

https://drive.google.com/uc?export=download&id=1ffGDQfisrZAHwIOz83DKx9I7ZyzigeA9

### Reproduce Training on APTOS

```bash
torchrun main_finetune.py --lr 1.25e-5 --blr 5e-5 --warmup_epochs 10 --task /path/to/output/dir --fix_backbone False --data_path /path/to/aptos 
```


### Citation

If you find this repository useful, please consider citing this paper:
```
@misc{zoellin2024blockexpandeddinoretadapting,
      title={Block Expanded DINORET: Adapting Natural Domain Foundation Models for Retinal Imaging Without Catastrophic Forgetting}, 
      author={Jay Zoellin and Colin Merk and Mischa Buob and Amr Saad and Samuel Giesser and Tahm Spitznagel and Ferhat Turgut and Rui Santos and Yukun Zhou and Sigfried Wagner and Pearse A. Keane and Yih Chung Tham and Delia Cabrera DeBuc and Matthias D. Becker and Gabor M. Somfai},
      year={2024},
      eprint={2409.17332},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2409.17332}, 
}
```


