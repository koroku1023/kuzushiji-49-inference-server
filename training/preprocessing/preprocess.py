import albumentations as albu
from albumentations.pytorch import transforms as AT


def img_transformer(
    image_size=(224, 224), mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0, 5]
):

    return albu.Compose(
        [
            albu.Resize(image_size[0], image_size[1]),
            albu.Normalize(mean=mean, std=std),
            AT.ToTensorV2(),
        ]
    )
