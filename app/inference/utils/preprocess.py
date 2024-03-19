from typing import List, Tuple, Union

import numpy as np
import albumentations as albu
from albumentations.pytorch import transforms as AT


def img_transformer(
    image_size: Union[List[int], Tuple[int]] = (224, 224),
    mean: List[float] = [0.5, 0.5, 0.5],
    std: List[float] = [0.5, 0.5, 0.5],
):

    transformer = albu.Compose(
        [
            albu.Resize(image_size[0], image_size[1]),
            albu.Normalize(mean=mean, std=std),
            AT.ToTensorV2(),
        ]
    )

    return transformer
