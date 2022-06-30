from torchvision import transforms
from PIL import Image
from plotly import express as px
import numpy as np
import pandas as pd


def plot_image(img, unnormalize=True):

    if unnormalize:
        inv_normalize = transforms.Normalize(
            mean=[-0.4914 / 0.2023, -0.4822 / 0.1994, -0.4465 / 0.2010],
            std=[1 / 0.2023, 1 / 0.1994, 1 / 0.2010],
        )
        img = inv_normalize(img)
    img = np.uint8(img.numpy().transpose(2, 1, 0).transpose(1, 0, 2) * 255)
    px.imshow(img, width=300, height=300).show()
