import torch
import matplotlib.pyplot as plt
import datasets.ss_transforms as sstr
import numpy as np

from PIL import Image


def show_output(model, image, label):
    model.eval()
    with torch.no_grad():
        output = model(image.unsqueeze(0))['out']
        pred = torch.argmax(output, dim=1).squeeze(0)

    unorm = sstr.UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

    fig, ax = plt.subplots(1, 3, dpi=250)
    fig.subplots_adjust(wspace=0, hspace=0)
    ax[0].imshow(unorm(image).permute(1, 2, 0))
    ax[0].set_axis_off()
    ax[1].imshow(labels_to_pil_image(label))
    ax[1].set_axis_off()
    ax[2].imshow(labels_to_pil_image(pred))
    ax[2].set_axis_off()

    return fig


def labels_to_pil_image(tensor):

    # Color palette for segmentation masks
    PALETTE = np.array(
        [
            [128,64,128],
            [244,35,232],
            [70,70,70],
            [102,102,156],
            [190,153,153],
            [153,153,153],
            [250,170,30],
            [220,220,0],
            [107,142,35],
            [152,251,152],
            [70,130,180],
            [220,20,60],
            [255,0,0],
            [0,0,142],
            [0,0,230],
            [119,11,32]
        ]
        + [[0, 0, 0] for i in range(256 - 16)],
        dtype=np.uint8,
    )
    pil_out = Image.fromarray(tensor.numpy().astype(np.uint8), mode='P')
    pil_out.putpalette(PALETTE)
    return pil_out