from diffusers.utils.loading_utils import load_image
from PIL import Image, UnidentifiedImageError
from threading import Thread
import config, utils
import numpy as np
import traceback
import discord
import logging
import asyncio
import time
import io
import os

NAME = "upscale"
DESCRIPTION = "Upscales an image. If message ID and url is empty, the last message in the channel will be used."
ARGUMENTS = {
    "url": "An URL to an image.",
    "message_id": "An ID of an message containing an image.",
    "file": "An image file.",
    "upscale": "The amount to upscale the image.",
}

logger = logging.getLogger("main")
esrgan = None


def init():
    global esrgan

    # load ESRGAN model
    if config.ESRGAN_MODEL:
        esrgan = utils.load_esrgan_model(config.ESRGAN_MODEL, config.DEVICE)


@discord.app_commands.describe(
    url="An URL to an image.",
    message_id="An ID of an message containing an image.",
    file="An image file.",
    upscale="The amount to upscale the image.",
)
async def command(
    interaction: discord.Interaction,
    url: str = None,
    message_id: str = None,
    file: discord.Attachment = None,
    upscale: float = None,
):
    if config.ESRGAN_MODEL is None:
        return await interaction.response.send_message(
            "Upscaling is disabled on this bot.", ephemeral=True
        )

    try:
        if message_id is None and url is None and file is None:
            message = await anext(interaction.channel.history(limit=1))
            if message is None:
                raise ValueError("No messages in channel.")

            image, attachment = await utils.load_image_from_message(message)
        else:
            image, attachment = await utils.load_image_from_args(
                message_id, url, file, interaction.channel
            )
    except ValueError as e:
        return await interaction.response.send_message(str(e), ephemeral=True)

    await interaction.response.send_message("Upscaling...")
    logger.info(f"Upscaling image on message {message_id}...")

    def worker(loop):
        nonlocal file

        start_time = time.time()
        np_image = np.array(image)

        try:
            output, _ = esrgan.enhance(np_image, outscale=upscale)
        except RuntimeError as e:
            logger.error(e)
            return utils.edit(
                loop,
                interaction,
                utils.error("Too much memory allocated. ping me ill fix it later"),
            )
        except Exception as e:
            traceback.print_exc()
            return utils.edit(loop, interaction, utils.error(e))

        outpath = os.path.join(config.UPSCALED_DIR, f"{interaction.id}.png")
        upscaled = Image.fromarray(output)
        upscaled.save(outpath)

        # create embed
        embed = discord.Embed(
            title=f"Upscaled Image (Total Time: {time.time() - start_time:.2f}s)",
            color=config.PRIMARY_EMBED_COLOR,
        )
        embed.set_author(
            name=interaction.user.display_name, icon_url=interaction.user.avatar.url
        )
        embed.set_image(url="attachment://upscaled.png")
        utils.add_fields(
            embed,
            {
                "Model": esrgan.model_name,
                "Upscale": f"x{upscale or esrgan.scale:g}",
                "Size": f"{upscaled.width}x{upscaled.height}",
                "Original Size": f"{image.width}x{image.height}",
            },
        )

        dfile = discord.File(
            outpath,
            "upscaled.png",
            spoiler=(
                attachment.is_spoiler()
                if message_id or ((url is None) and (file is None))
                else False
            ),
        )
        utils.edit(loop, interaction, embed=embed, attachments=[dfile])

        logger.info(f"Finished upscaling image on message {message_id}.")

    thread = Thread(target=worker, args=(asyncio.get_event_loop(),))
    thread.start()
