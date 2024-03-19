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


def under_sampling(
    images: np.array, labels: np.array, num_classes: int, threshold: int = 2000
):

    sampled_images = []
    sampled_labels = []

    for idx in range(num_classes):

        class_images = images[labels == idx]
        class_labels = labels[labels == idx]

        if len(class_images) >= threshold:
            num_sampled = int(len(class_images) * 0.6)
            sampled_indices = np.random.choice(
                range(len(class_images)),
                num_sampled,
                replace=False,
            )
            sampled_images.append(class_images[sampled_indices])
            sampled_labels.append(class_labels[sampled_indices])
        else:
            sampled_images.append(class_images)
            sampled_labels.append(class_labels)

    sampled_images = np.concatenate(sampled_images, axis=0)
    sampled_labels = np.concatenate(sampled_labels, axis=0)

    return sampled_images, sampled_labels
