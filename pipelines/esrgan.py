from realesrgan import RealESRGANer
from PIL import Image
import numpy as np
import io


def upscale(model: RealESRGANer, image_bytes: bytes, outscale: int = None):
    image = np.array(Image.open(io.BytesIO(image_bytes)))

    try:
        output, _ = model.enhance(image, outscale=outscale)
    except RuntimeError:
        raise RuntimeError("Too much memory allocated. ping me ill fix it later")

    return Image.fromarray(output)
