import os
from typing import Any
import numpy as np
from PIL import Image
from torch import from_numpy
from torchvision.datasets import VisionDataset
import datasets.ss_transforms as tr

class_map = {
    1: 13,   # ego_vehicle : vehicle
    7: 0,    # road
    8: 1,    # sidewalk
    11: 2,   # building
    12: 3,   # wall
    13: 4,   # fence
    17: 5,   # pole
    18: 5,   # poleGroup: pole
    19: 6,   # traffic light
    20: 7,   # traffic sign
    21: 8,   # vegetation
    22: 9,   # terrain
    23: 10,  # sky
    24: 11,  # person
    25: 12,  # rider
    26: 13,  # car : vehicle
    27: 13,  # truck : vehicle
    28: 13,  # bus : vehicle
    32: 14,  # motorcycle
    33: 15,  # bicycle
}


class GTADataset(VisionDataset):

    def __init__(self,
                 root: str,
                 list_samples: list[str],
                 transform: tr.Compose = None):
        super().__init__(root=root, transform=transform, target_transform=None)
        self.list_samples = list_samples
        self.target_transform = self.get_mapping()
        self.return_unprocessed_image = False
        self.style_tf_fn = None

    # maps the classes of the gta5 into the ones given for the idda
    @staticmethod
    def get_mapping():
        mapping = np.zeros((256,), dtype=np.int64) + 255
        for k, v in class_map.items():
            mapping[k] = v
        return lambda x: from_numpy(mapping[x])

    # set style transfer function
    def set_style_tf_fn(self, style_tf_fn):
        self.style_tf_fn = style_tf_fn

    def __getitem__(self, index: int) -> Any:
        # get the id_code of the given index
        id_code = self.list_samples[index]
        # get image path
        img_path = os.path.join(self.root, 'images', f'{id_code}')
        image = Image.open(img_path)
        # get label path
        lbl_path = os.path.join(self.root, 'labels', f'{id_code}')
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
            image, label = self.transform(image, lbl=label)
        label = self.target_transform(label)

        return image, label

    def __len__(self) -> int:
        return len(self.list_samples)
