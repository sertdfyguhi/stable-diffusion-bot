from diffusers.utils.loading_utils import load_image
from PIL import Image, UnidentifiedImageError
import discord
import config
import utils

NAME = "img2img"
DESCRIPTION = "Generates an image using img2img."
ARGUMENTS = {
    "model": "The model to use. /models to list all available models.",
    "image_url": "An URL to an image.",
    "prompt": "The text that guides the image generation.",
    "negative_prompt": "The text that specifies what you don't want to see in the generated image.",
    "guidance_scale": "A number that specifies how closely the image should follow your prompt.",
    "step_count": "The number of sampling iterations in the generation process.",
    "seed": "The number used to initialize the image generation. Used to reproduce images.",
    "downscale_factor": "The amount to downscale the base image.",
}


def handle(
    interaction: discord.Interaction,
    model: str,
    image_url: str,
    prompt: str,
    negative_prompt: str = None,
    guidance_scale: float = 8.0,
    step_count: int = 14,
    seed: str = None,
    downscale_factor: float = 1.0,
):
    if seed and not seed.isnumeric():
        return ValueError("Seed must be a number.")

    try:
        image = load_image(image_url)
    except UnidentifiedImageError:
        raise ValueError("URL response could not be read as an image.")
    except ValueError:
        raise ValueError("Invalid URL.")

    # downscale image by factor
    new_width = int(image.size[0] / downscale_factor)
    new_height = int(image.size[1] / downscale_factor)
    image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    if image.width > config.MAX_WIDTH:
        raise ValueError(
            f"Image width ({image.size[0]}x{image.size[1]}) cannot be higher than {config.MAX_WIDTH} pixels."
        )
    elif image.height > config.MAX_HEIGHT:
        raise ValueError(
            f"Image height ({image.size[0]}x{image.size[1]}) cannot be higher than {config.MAX_HEIGHT} pixels."
        )

    return utils.Img2ImgGenerationRequest(
        interaction,
        model,
        prompt,
        negative_prompt,
        guidance_scale,
        step_count,
        int(seed) if seed else None,
        image,
    )
