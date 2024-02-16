from realesrgan import RealESRGANer
from PIL import Image
import numpy as np


def upscale(model: RealESRGANer, image: Image.Image, outscale: int = None):
    image = np.array(image)

    try:
        output, _ = model.enhance(image, outscale=outscale)
    except RuntimeError:
        raise RuntimeError("Too much memory allocated. ping me ill fix it later")

    return Image.fromarray(output)
