# SAM-Swin: SAM-Driven Dual-Swin Transformers with Adaptive Lesion Enhancement for Laryngo-Pharyngeal Tumor Detection

This repo is the official implementation of [SAM-Swin: SAM-Driven Dual-Swin Transformers with Adaptive Lesion Enhancement for Laryngo-Pharyngeal Tumor Detection](https://arxiv.org/abs/2410.21813).


## Fine-tune SAM2

To fine-tune SAM2 tailored for your tasks, we recommend following the guidelines provided in the original repository: [MedSAM2](https://github.com/bowang-lab/MedSAM/tree/MedSAM2)


## Dataset

Organize your datasets in the following manner:

```markdown
datasets/
├── dataset1/
│   ├── global/
│   │   ├── train/
│   │   │   ├── benign/
│   │   │   ├── normal/
│   │   │   └── tumor/
│   │   ├── val/
│   │   │   ├── benign/
│   │   │   ├── normal/
│   │   │   └── tumor/
│   │   └── test/
│   │       ├── benign/
│   │       ├── normal/
│   │       └── tumor/
│   └── local_seg/
│       ├── train/
│       │   ├── benign/
│       │   ├── normal/
│       │   └── tumor/
│       ├── val/
│       │   ├── benign/
│       │   ├── normal/
│       │   └── tumor/
│       └── test/
│           ├── benign/
│           ├── normal/
│           └── tumor/
├── dataset6/
│   └── ...
```

## Training

We train the SAM-Swin in two stages.

1. Stage 1, run:

   ```bash
   python -m torch.distributed.launch --nproc_per_node 2 --master_port 12345 main.py --cfg configs/dynamic.yaml --batch-size 32 --pretrained swinv2_base_patch4_window16_256.pth --cache-mode full --amp-opt-level O1 --accumulation-steps 4 --fused_window_process --fused_layernorm --tag exp
   ```

2. Stage 2, run:

   ```bash
   python -m torch.distributed.launch --nproc_per_node 2 --master_port 12345 main.py --cfg configs/ft_baseline.yaml --batch-size 32 --pretrained <Your path of latest checkpoint at the Stage 1> --cache-mode full --amp-opt-level O1 --accumulation-steps 4 --fused_window_process --fused_layernorm --tag exp_ft
   ```


## Testing

Using DDP, Run:

```
python -m torch.distributed.launch --nproc_per_node 2 --master_port 12345 main.py --cfg configs/dynamic.yaml --resume <Your path of the checkpoint> --cache-mode full --amp-opt-level O1 --accumulation-steps 4 --fused_window_process --fused_layernorm --tag exp --eval
```


## Acknowledgement

The code of SAM-Swin is built upon [MedSAM2](https://github.com/bowang-lab/MedSAM/tree/MedSAM2) and [Swin Transformer](https://github.com/microsoft/Swin-Transformer?tab=readme-ov-file), and we express our gratitude to these awesome projects.


## Citing SAM-Swin

```
@misc{wei2024samswinsamdrivendualswintransformers,
      title={SAM-Swin: SAM-Driven Dual-Swin Transformers with Adaptive Lesion Enhancement for Laryngo-Pharyngeal Tumor Detection}, 
      author={Jia Wei and Yun Li and Xiaomao Fan and Wenjun Ma and Meiyu Qiu and Hongyu Chen and Wenbin Lei},
      year={2024},
      eprint={2410.21813},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2410.21813}, 
}
```
