from PIL import Image, UnidentifiedImageError
import requests
import discord
import config
import utils
import io

NAME = "img2img"
DESCRIPTION = "Generates an image using img2img."


def handle(
    interaction: discord.Interaction,
    model: str,
    image_url: str,
    prompt: str,
    negative_prompt: str = "",
    guidance_scale: float = 8.0,
    step_count: int = 14,
    seed: int = None,
):
    image_bytes = requests.get(image_url, headers={"Accept": "image/*"}).content

    try:
        image = Image.open(io.BytesIO(image_bytes))
    except UnidentifiedImageError:
        raise ValueError("URL response could not be read as an image.")

    if image.width > config.MAX_WIDTH:
        raise ValueError(
            f"Image width cannot be higher than {config.MAX_WIDTH} pixels."
        )
    elif image.height > config.MAX_HEIGHT:
        raise ValueError(
            f"Image height cannot be higher than {config.MAX_HEIGHT} pixels."
        )

    return utils.Img2ImgGenerationRequest(
        interaction,
        model,
        prompt,
        negative_prompt,
        guidance_scale,
        step_count,
        seed,
        image,
    )
