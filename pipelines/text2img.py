import discord
import config
import utils

NAME = "text2img"
DESCRIPTION = "Generates an image using text2img."


def handle(
    interaction: discord.Interaction,
    model: str,
    prompt: str,
    negative_prompt: str = "",
    guidance_scale: float = 8.0,
    step_count: int = 14,
    seed: str = None,
    width: int = 512,
    height: int = 680,
):
    # validate parameters
    if width > config.MAX_WIDTH:
        return ValueError(
            f"Width ({width}) cannot be higher than {config.MAX_WIDTH} pixels."
        )
    elif height > config.MAX_HEIGHT:
        return ValueError(
            f"Height ({height}) cannot be higher than {config.MAX_HEIGHT} pixels."
        )
    elif seed and not seed.isnumeric():
        return ValueError("Seed must be a number.")

    return utils.Text2ImgGenerationRequest(
        interaction,
        model,
        prompt,
        negative_prompt,
        guidance_scale,
        step_count,
        int(seed) if seed else None,
        width,
        height,
    )
