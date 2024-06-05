# DifAttack++
The official code for the paper titled as "DifAttack++: Query-Efficient Black-Box Adversarial Attack via Hierarchical Disentangled Feature Space in Cross Domain". It will be available soon in arxiv.

Our previous conference version called "DifAttack: Query-Efficient Black-Box Attack via Disentangled Feature Space" has been accepted by AAAI 2024.
The supplementary file is available at our [Arxiv](https://arxiv.org/abs/2309.14585) version.

![Overview](https://github.com/csjunjun/DifAttack/blob/master/Architecture.jpeg)


## Setup
Please download the test set and model weights of DifAttack++ from [GoogleDrive](https://drive.google.com/drive/folders/1gCOxEwJGPO_tKKLPldRsRFCAtgOM40K5?usp=sharing).
The model weights for DifAttack can be downloaded from [GoogleDrive](https://drive.google.com/drive/folders/1o4yPWxAC575PT_mQSxV4d7BCLCbC2oRV?usp=sharing).

## Train autoencoders for image reconstruction and feature disentanglement:
set mode="train" in main.py
```
Python main.py
```

## Perform black-box attack
set mode="test" in main.py
```
Python main.py
```

## Acknowledgements
Part of the code is partially derived from ImageReconstruction [Github](https://github.com/SikanderBinMukaram/ImageReconstructionAutoEncoder/blob/main/ImageReconstruction.ipynb) and torchattacks [Github](https://github.com/Harry24k/adversarial-attacks-pytorch/tree/master).


## Citation
If you find this work useful for your research, you can cite:
```
@inproceedings{JunDifAttack2024,
title={DifAttack: Query-Efficient Black-Box Attack via Disentangled Feature Space},
author={Liu, Jun and Zhou, Jiantao and Zeng, Jiandian and Tian, Jinyu},
booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
volume={38},
number={4}, 
pages={3666-3674} ,
year={2024}, 
month={Mar.}, 
url={https://ojs.aaai.org/index.php/AAAI/article/view/28156}, 
DOI={10.1609/aaai.v38i4.28156}
}```
