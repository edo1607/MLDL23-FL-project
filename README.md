# Application of federated learning and domain adaptation to semantic segmentation

Implementation of an algorithm for an application of federated learning and domain adaptation to semantic segmentation

**Authors**: Andrea Ruglioni, Giulia Monchietto, Edoardo Venturini

## Summary

In this work we started from a centralized and supervised baseline to implement a model for semantic segmentation, using the DeepLabV3 network with MobileNetV2 as the backbone. Subsequently, we introduced Federated Learning (FL) to harness training data from distributed devices while upholding privacy preservation; we tested this model on the IDDA dataset. To address the scarcity of annotated datasets in real-world scenarios, we then implemented a Semi-Supervised Learning (SSL) and an Adversarial Learning (AL) frameworks. Moreover, we tried a Fourier Domain Adaptation (FDA) approach to mitigate the performance gap between the source and target datasets. In the semi-supervised setting, the algorithm was tested employing GTA5 as the source dataset and IDDA as the target dataset. The experiments we did underline a good behaviour of the algorithm in the supervised framework; on the contrary, the semi-supervised model can be noticeably improved.

## Setup

### Preliminary operations

1) Clone this repository.
2) Move to the root path of your local copy of the repository.
3) Create a new conda virtual environment and activate it:
```
conda env create -f mldl23fl.yml
conda activate Project
```

### Datasets

1) Download the the IDDA dataset from [here](https://mega.nz/file/yBwVGR6A#z2AyGYdsuHERRY67i6JKxhK9cbgVwhYWp4PyrrITbMQ).
2) Download the GTA5 dataset from [here](https://mega.nz/file/ERkiQBaY#h-wktK7U7MpIG5nf-rMWF7d76NEM5ae_MrAmELftNR0).
5) Extract all the datasets' archives. Then, move the datasets' folders in ```data/[dataset_name]```, 
where ```[dataset_name]``` is one of ```{IDDA, GTA5}```.

### Experiments' logging

Make a new [wandb](https://wandb.ai/site) account if you do not have one yet, and create a new wandb project.

### How to run

Create a jupyter notebook file and upload it in the "run_project" folder. Then import the ```main.py ``` file and type ```!main.py``` in a new chunk followed by the line arguments, just like ```--num_epochs 9```. Pay attention while setting these parameters, that are a little bit different depending on which part of the model you would like to use.
