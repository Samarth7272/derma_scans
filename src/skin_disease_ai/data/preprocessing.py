import io
from PIL import Image
import numpy as np
import cv2
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def preprocess_rgb(image, apply_clahe=False, remove_hair=False, denoise=False):
    if isinstance(image, np.ndarray):
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        image = Image.fromarray(image)
    elif isinstance(image, Image.Image):
        pass
    elif isinstance(image, bytes):
        image = Image.open(io.BytesIO(image))
    elif isinstance(image, str) or hasattr(image, "read"):
        image = Image.open(image)
    else:
        raise TypeError(f"Unsupported image input type: {type(image)}")

    image = image.convert("RGB")
    arr = np.array(image)

    if remove_hair:
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        _, mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
        arr = cv2.inpaint(arr, mask, 1, cv2.INPAINT_TELEA)

    if denoise:
        arr = cv2.fastNlMeansDenoisingColored(arr, None, 5, 5, 7, 21)

    if apply_clahe:
        lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        arr = cv2.cvtColor(cv2.merge((clahe.apply(l), a, b)), cv2.COLOR_LAB2RGB)

    return Image.fromarray(arr)


def build_transforms(image_size=224, split="test", augmentation="none", normalize=True):
    if isinstance(image_size, dict):
        image_size = image_size.get("image_size", 224)
    if not isinstance(image_size, int):
        image_size = 224

    tfms = [transforms.Resize((image_size, image_size)), transforms.ToTensor()]
    if normalize:
        tfms.append(transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD))
    return transforms.Compose(tfms)