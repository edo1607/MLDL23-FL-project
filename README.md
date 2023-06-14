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

Run the code written in the ```test.ipynb``` file. The key parameter you must choose is ```--step```, that decides the part of the model to employ. More specifically, type:

- ```--step 1``` if you would like to train and test the algorithm on IDDA in a supervised centralized environment;
- ```--step 2``` if you would like to train and test the algorithm on IDDA in a supervised federated environment;
- ```--step 3a``` if you would like to train the algorithm on GTA5 and test it on IDDA in a centralized environment;
- ```--step 3b``` if you would like to train the algorithm on GTA5 and test it on IDDA in a centralized setting, using the style-transfer technique;
- ```--step 4``` if you would like to run the algorithm in a federated and semi-supervised setting, training a teacher model on GTA5 and a student model on IDDA and using pseudo-labels as ground truth instead of the original labels;
- ```--step 5a``` if you would like to run the algorithm in a centralized environment utilizing adversarial learning;
- ```--step 5b``` if you would like to run the algorithm in a federated environment utilizing adversarial learning.
- ```--step 5c``` if you would like to run the algorithm in a federated and semi-supervised setting utilizing adversarial learning.

Step 4 also require ```--path_model``` to be set, that represents the checkpoint of step 3 which the experiment is based on.
Similarly does the step 5c, requiring ```--path_model``` and ```--path_discriminator```.

In general, feel free to set the line arguments however you prefer; in ```test.ipynb``` file, the only argument we set is ```--step 1``` as an example. Besides necessary values, if you do not choose any value for a parameter, it will be set to a certain default value, as in ```utils/args.py```, where are shown all the possible hyper-parameters.
