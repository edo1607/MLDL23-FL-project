import os
from typing import Any
import numpy as np
from PIL import Image
from torch import from_numpy
from torchvision.datasets import VisionDataset
import datasets.ss_transforms as tr

class_eval = [255, 2, 4, 255, 11, 5, 0, 0, 1, 8, 13, 3, 7, 6, 255, 255, 15, 14, 12, 9, 10]


class IDDADataset(VisionDataset):

    def __init__(self,
                 root: str,
                 list_samples: list[str],
                 transform: tr.Compose = None,
                 client_name: str = None):
        super().__init__(root=root, transform=transform, target_transform=None)
        self.list_samples = list_samples
        self.client_name = client_name
        self.target_transform = self.get_mapping()
        self.return_unprocessed_image = False
        self.style_tf_fn = None

    # maps a grey scale into the 16 considered classes during evaluation
    # plus the 255 which gathers all the not labeled objects
    # this is because the idda dataset has 24 classes,
    # but only 16 of them are taken into account for evaluation
    @staticmethod
    def get_mapping():
        classes = class_eval
        mapping = np.zeros((256,), dtype=np.int64) + 255
        for i, cl in enumerate(classes):
            mapping[i] = cl
        return lambda x: from_numpy(mapping[x])

    def set_style_tf_fn(self, style_tf_fn):
        self.style_tf_fn = style_tf_fn

    def __getitem__(self, index: int) -> Any:
        # get the id_code of the given index
        id_code = self.list_samples[index]
        # get image path
        img_path = os.path.join(self.root, 'images', f'{id_code}.jpg')
        image = Image.open(img_path)
        # get label path
        lbl_path = os.path.join(self.root, 'labels', f'{id_code}.png')
        label = Image.open(lbl_path)
        
        # return just the unprocessed image
        # it is used to extract the dataset's style
        if self.return_unprocessed_image:
            return image
        # apply style transfer function if present
        if self.style_tf_fn is not None:
            image = self.style_tf_fn(image)
        # transform both image and label in order to have same crops etc.
        # note that transformers like Normalize will not be applied to the label
        if self.transform:
            image,label = self.transform(image, lbl=label)
        label = self.target_transform(label)

        return image, label

    def __len__(self) -> int:
        return len(self.list_samples)
