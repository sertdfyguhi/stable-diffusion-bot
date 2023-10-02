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
    seed: int = None,
    width: int = 512,
    height: int = 680,
):
    # validate parameters
    if width > config.MAX_WIDTH:
        return utils.error(f"Width cannot be higher than {config.MAX_WIDTH} pixels.")
    elif height > config.MAX_HEIGHT:
        return utils.error(f"Height cannot be higher than {config.MAX_HEIGHT} pixels.")

    return utils.Text2ImgGenerationRequest(
        interaction,
        model,
        prompt,
        negative_prompt,
        guidance_scale,
        step_count,
        seed,
        width,
        height,
    )
