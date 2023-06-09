import numpy as np
import random
from PIL import Image
import cv2
from tqdm import tqdm
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
import json
import os


class StyleAugment:

    def __init__(self, n_images_per_style=25, L=None, size=(1920, 1080), b=None):
        self.styles = []
        self.styles_names = []
        self.n_images_per_style = n_images_per_style
        self.L = L
        self.size = size
        self.sizes = None
        self.b = b

    # transform the input from PIL image to numpy array
    def preprocess(self, x):
        x = x.resize(self.size, Image.BICUBIC)
        x = np.asarray(x, np.float32)
        x = x[:, :, ::-1]
        x = x.transpose((2, 0, 1))
        return x.copy()

    # transform the input from numpy array to PIL image
    def deprocess(self, x, size):
        x = Image.fromarray(np.uint8(x).transpose((1, 2, 0))[:, :, ::-1])
        x = x.resize(size, Image.BICUBIC)
        return x

    def add_style(self, dataset, name=None):

        if name is not None:
            self.styles_names.append(name)

        dataset.return_unprocessed_image = True
        n = 0
        styles = []

        for sample in dataset:

            image = self.preprocess(sample)

            if n >= self.n_images_per_style:
                break
            styles.append(self._extract_style(image))
            n += 1

        if self.n_images_per_style > 1:
            styles = np.stack(styles, axis=0)
            style = np.mean(styles, axis=0)
            self.styles.append(style)
        elif self.n_images_per_style == 1:
            self.styles += styles

        dataset.return_unprocessed_image = False

    def _extract_style(self, img_np):
        fft_np = np.fft.fft2(img_np, axes=(-2, -1))
        amp = np.abs(fft_np)
        amp_shift = np.fft.fftshift(amp, axes=(-2, -1))
        if self.sizes is None:
            self.sizes = self.compute_size(amp_shift)
        h1, h2, w1, w2 = self.sizes
        style = amp_shift[:, h1:h2, w1:w2]
        return style

    # compute the dimensions of the center region mask
    def compute_size(self, amp_shift):
        _, h, w = amp_shift.shape
        b = (np.floor(np.amin((h, w)) * self.L)).astype(int) if self.b is None else self.b
        c_h = np.floor(h / 2.0).astype(int)
        c_w = np.floor(w / 2.0).astype(int)
        h1 = c_h - b
        h2 = c_h + b + 1
        w1 = c_w - b
        w2 = c_w + b + 1
        return h1, h2, w1, w2

    def apply_style(self, img):

        if len(self.styles) > 0:
            n = random.randint(0, len(self.styles) - 1)
            style = self.styles[n]
        else:
            style = self.styles[0]

        W, H = img.size
        img_np = self.preprocess(img)

        fft_np = np.fft.fft2(img_np, axes=(-2, -1))
        amp, pha = np.abs(fft_np), np.angle(fft_np)
        amp_shift = np.fft.fftshift(amp, axes=(-2, -1))
        h1, h2, w1, w2 = self.sizes
        amp_shift[:, h1:h2, w1:w2] = style
        amp_ = np.fft.ifftshift(amp_shift, axes=(-2, -1))

        fft_ = amp_ * np.exp(1j * pha)
        img_np_ = np.fft.ifft2(fft_, axes=(-2, -1))
        img_np_ = np.real(img_np_)
        img_np__ = np.clip(np.round(img_np_), 0., 255.)

        img_with_style = self.deprocess(img_np__, (W, H))

        return img_with_style

    def show(self, img_trg, img_src):
      
        W_trg, H_trg = img_trg.size
        img_trg = self.preprocess(img_trg)

        W_src, H_src = img_src.size
        img_src = self.preprocess(img_src)

        self.size = (max(W_src, W_trg), max(H_src, H_trg))

        style = self._extract_style(img_trg)

        fft_np = np.fft.fft2(img_src, axes=(-2, -1))
        amp, pha = np.abs(fft_np), np.angle(fft_np)
        amp_shift = np.fft.fftshift(amp, axes=(-2, -1))
        h1, h2, w1, w2 = self.sizes
        amp_shift[:, h1:h2, w1:w2] = style
        amp_ = np.fft.ifftshift(amp_shift, axes=(-2, -1))

        fft_ = amp_ * np.exp(1j * pha)
        img_src_ = np.fft.ifft2(fft_, axes=(-2, -1))
        img_src_ = np.real(img_src_)
        img_src__ = np.clip(np.round(img_src_), 0., 255.)

        img_trg = self.deprocess(img_trg, self.size)
        img_src = self.deprocess(img_src, self.size)
        img_with_style = self.deprocess(img_src__, self.size)

        fig, ax = plt.subplots(1, 3, dpi=250)
        fig.subplots_adjust(wspace=0, hspace=0)
        ax[0].imshow(img_trg)
        ax[0].set_axis_off()
        ax[1].imshow(img_with_style)
        ax[1].set_axis_off()
        ax[2].imshow(img_src)
        ax[2].set_axis_off()

        return fig

    def create_bank_styles(self, clients, path='styles'):
        for c in tqdm(clients, total=len(clients)):
            self.add_style(c.dataset, c.name)
        
        with open(os.path.join('data/idda/bank', f'{path}_{self.L}.json'), 'w') as f:
            data = {
              'styles': {name: style.tolist() for name, style in zip(self.styles_names, self.styles)},
              'sizes': np.array(self.sizes).tolist()
            }
            json.dump(data, f, indent=4)

    def load_bank_styles(self, L=None, path='styles'):
        try:
            with open(os.path.join('data/idda/bank', f'{path}_{L}.json'), 'r') as f:
                data = json.load(f)
                for name, style in data['styles'].items():
                    self.styles_names.append(name)
                    self.styles.append(np.array(style))

                self.sizes = data['sizes']

                return True
        except OSError:
            return False


