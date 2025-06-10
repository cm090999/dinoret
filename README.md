This repository is based on the implementation from RETFound: [Article](https://www.nature.com/articles/s41586-023-06555-x), [Github](https://github.com/rmaphoh/RETFound_MAE.git)

## BE-DINORET: Block Expanded DINOv2 for Retinal Imaging
This is the official repository for [Block Expanded DINORET: Adapting Natural Domain Foundation Models for Retinal Imaging Without Catastrophic Forgetting](https://arxiv.org/abs/2409.17332).

### Self-Supervised pretraining on retinal images
For running the pretraining for DINOv2 and the block expanded version, please refer to the official [DINOv2 repository](https://github.com/facebookresearch/dinov2.git) as well as the [official implementation of the block expansion](https://github.com/TencentARC/LLaMA-Pro.git).

### Key features

- Finetuning scripts for BE-DINOv2 and DINOv2 for retinal images
- RETFound finetuning and inference scripts

## Installing the Environment

**System Requirements:** Ubuntu 22.04 LTS

### Installation Steps

1. **Create Environment**
   
   Choose the appropriate setup script based on your hardware:
   
   - **Standard setup:**
     ```bash
     bash scripts/create_env.sh
     ```
   
   - **GH200 compatibility:**
     ```bash
     bash scripts/create_env_gh200.sh
     ```
     > Note: Use this alternative if you're developing on a GH200, which is incompatible with the standard torch version.

2. **Activate Environment**
   ```bash
   source .env/bin/activate
   ```

## Available Datasets

There is a dataloader that support the following dataset in their respective original structure after download: [aptos](https://www.kaggle.com/competitions/aptos2019-blindness-detection) [eyepacs](https://www.kaggle.com/competitions/diabetic-retinopathy-detection/data), [idrid](https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid), [messidor](https://www.adcis.net/en/third-party/messidor2/), [drtid](https://github.com/fdu-vts/drtid), [papila](https://figshare.com/articles/dataset/PAPILA/14798004/1?file=28454352)

## Pretrained Weights

We provide the pre-trained encoder weights of our domain-adapted models on Huggingface: [`cmerk/dinoret`](https://huggingface.co/cmerk/dinoret). We provide weights for DINORET, BE-DINORET and LoRA-DINORET. 

 - DINORET: ```"hf:cmerk/dinoret:dinoret.pth"```
 - BE-DINORET: ```"hf:cmerk/dinoret:bedinoret.pth"```
 - LoRA-DINORET: ```"hf:cmerk/dinoret:loradinoret.pth"```


## Running Experiments


**Finetuning experiments on APTOS:**

- **DINOv2 ViT-b** - Baseline DINOv2 ViT-b
```bash
torchrun main_finetune.py --task /path/to/output/dir --data_path /path/to/aptos --lr 1.25e-5 --blr 5e-5 --warmup_epochs 10 --fix_backbone False --pretrained_checkpoint ''
```

- **DINORET** - Domain adapted DINOv2 ViT-b (DINORET)
```bash
torchrun main_finetune.py --task /path/to/output/dir --data_path /path/to/aptos --lr 1.25e-5 --blr 5e-5 --warmup_epochs 10 --fix_backbone False --pretrained_checkpoint hf:cmerk/dinoret:dinoret.pth
```

- **BE-DINORET** - Domain adapted DINOv2 ViT-b using block expansion (BE-DINORET)
```bash
torchrun main_finetune.py --task /path/to/output/dir --data_path /path/to/aptos --lr 1.25e-5 --blr 5e-5 --warmup_epochs 10 --fix_backbone False --block_expansion_positions "3 7 11" --pretrained_checkpoint hf:cmerk/dinoret:bedinoret.pth
```

- **LoRA-DINORET** - Domain adapted DINOv2 ViT-b using LoRA (LoRA-DINORET)
```bash
torchrun main_finetune.py --task /path/to/output/dir --data_path /path/to/aptos --lr 1.25e-5 --blr 5e-5 --warmup_epochs 10 --fix_backbone False --lora_adaptation True --pretrained_checkpoint hf:cmerk/dinoret:loradinoret.pth
```

**Key Arguments**

| Argument | Description | Default |
|----------|-------------|---------|
| `--data_path` | Path to the dataset | Required |
| `--task` | Output location | Required |
| `--fix_backbone` | Whether the encoder backbone weights are frozen or not | True |
| `--block_expansion_positions` | Block Positions of the expanded Blocks | None |
| `--lora_adaptation` | Enable LoRA adaptation for training | False |
| `--model` | Model encoder architecture to use | dinov2_vitb14 |
| `--pretrained_checkpoint` | Path or HF URL with encoder weights **only encoder** | '' |

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


