# Decomposing Semantic Shifts for Composed Image Retrieval

🎉️ **This paper was accepted to AAAI 2024**

Composed image retrieval is a type of image retrieval task where the user provides a reference image as a starting point and specifies a text on how to shift from the starting point to the desired target image. However, most existing methods focus on the composition learning of text and reference images and oversimplify the text as a description, neglecting the inherent structure and the user's shifting intention of the texts. As a result, these methods typically take shortcuts that disregard the visual cue of the reference images.To address this issue, we reconsider the text as instructions and propose a Semantic Shift Network (SSN) that explicitly decomposes the semantic shifts into two steps: from the reference image to the visual prototype and from the visual prototype to the target image.Specifically, SSN explicitly decomposes the instructions into two components: degradation and upgradation, where the degradation is used to picture the visual prototype from the reference image, while the upgradation is used to enrich the visual prototype into the final representations to retrieve the desired target image.

![image](https://github.com/starxing-yuu/SSN/blob/master/overview.png)
---

## Environment Setup

Get the environment by

```
pip install -r requirement.txt
```

## Data Preparation

### CIRR

If you want to download the data, you can refer to [CIRR](https://github.com/Cuberick-Orion/CIRR). Our data is structed as below:

```
cirr_dataset/
│
├── cirr/
│   ├── captions
│   └── captions_ext
|   └── image_splits
│
├── dev/
│   ├── dev-0-0-img0.png
│   └── ......
│
|── images/
|   ├── train
|
└── test1/
    ├──test1-0-0-img0.png
    └──......
```

### Fashion-IQ

If you want to download the data, you can refer to [Fashion-IQ](https://github.com/XiaoxiaoGuo/fashion-iq). Our data is structed as below:

```
fashion-iq/
│
├── captions/
│   ├── cap.dress.train.json
|   └── cap.dress.val.json
│   └── ......
│
├── image_splits/
│   ├── split.dress.train.json
|   └── split.dress.val.json
│   └── ......
|
└── images/
    ├──245600258X.png
    └──......
```

## Train

```
python train_cirr.py 
--dataset CIRR 
--model ssn 
--project_name xxx 
--workspace xxx 
--projection_dim 512 
--hidden_dim 512 
--num_epochs 50 
--batch_size 128 
--lr 5e-5 
--lr_ratio 0.2 
--lr_gamma 0.1 
--lr_step_size 10 
--save_training 
--save_best 
--validation_frequency 1
```

---

## Citation

If you find our work useful, please cite our paper:

```
@inproceedings{yang2024decomposing,
title={Decomposing Semantic Shifts for Composed Image Retrieval},
author={Yang, Xingyu and Liu, Daqing and Zhang, Heng and Luo, Yong and Wang, Chaoyue and Zhang, Jing},
booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
volume={38},
number={7},
pages={6576--6584},
year={2024}
}
```

---

## Acknowledgement

We thank [CLIP4Cir](https://github.com/ABaldrati/CLIP4Cir) for their great codebase.

