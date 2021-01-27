import torch
import numpy as np
from PIL import Image
import os


def save_ensemble_image(image_numpy, image_flip_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_flip_pil = Image.fromarray(image_flip_numpy)
    image_flip_pil = image_flip_pil.transpose(Image.FLIP_LEFT_RIGHT)
    image_pil = np.asarray(image_pil).astype(np.uint16)
    image_flip_pil = np.asarray(image_flip_pil).astype(np.uint16)
    out = (image_flip_pil + image_pil) / 2
    out = out.astype(np.uint8)
    out = Image.fromarray(out)
    out.save(image_path)


def save_image(image_numpy, image_path, aspect_ratio=1.0):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """

    image_pil = Image.fromarray(image_numpy)
    h, w, _ = image_numpy.shape

    if aspect_ratio > 1.0:
        image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
    if aspect_ratio < 1.0:
        image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)
    image_pil.save(image_path)


def normimage(input_image, imtype=np.uint8):
    if isinstance(input_image, torch.Tensor):  # get the data from a variable
        image_tensor = input_image.data
        image_numpy = image_tensor[0].cpu().float().numpy()
        if image_numpy.shape[0] == 3:
            image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
            image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)