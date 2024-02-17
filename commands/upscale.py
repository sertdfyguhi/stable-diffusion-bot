from diffusers.utils.loading_utils import load_image
from PIL import Image, UnidentifiedImageError
import config, pipelines, utils
from threading import Thread
import traceback
import discord
import logging
import asyncio
import time
import io
import os

NAME = "upscale"
DESCRIPTION = "Upscales an image. If message ID and url is empty, the last message in the channel will be used."

logger = logging.getLogger("main")
esrgan = None


def init():
    global esrgan

    # load ESRGAN model
    if config.ESRGAN_MODEL:
        esrgan = utils.load_esrgan_model(config.ESRGAN_MODEL, config.DEVICE)


async def command(
    interaction: discord.Interaction,
    url: str = None,
    message_id: str = None,
):
    if config.ESRGAN_MODEL is None:
        return await interaction.response.send_message(
            "Upscaling is disabled on this bot.", ephemeral=True
        )

    if message_id or (url is None):
        if message_id:
            try:
                message = await interaction.channel.fetch_message(int(message_id))
            except ValueError:
                return await interaction.response.send_message(
                    "Message ID is not an ID.", ephemeral=True
                )
            except Exception:
                return await interaction.response.send_message(
                    f"Could not find message {message_id}!", ephemeral=True
                )
        else:
            message = await anext(interaction.channel.history(limit=1))
            if message is None:
                return await interaction.response.send_message(
                    "No messages in channel.", ephemeral=True
                )

        attachments = message.attachments
        if len(attachments) == 0:
            return await interaction.response.send_message(
                f"There are no attachments attached to the message.", ephemeral=True
            )

        try:
            attachment = next(
                attachment
                for attachment in attachments
                if attachment.content_type.startswith("image")
            )
        except StopIteration:
            return await interaction.response.send_message(
                f"There are no images attached to the message.", ephemeral=True
            )

        image = Image.open(io.BytesIO(await attachment.read()))
    else:
        try:
            image = load_image(url)
        except UnidentifiedImageError:
            return await interaction.response.send_message(
                "URL response could not be read as an image.", ephemeral=True
            )
        except ValueError:
            return await interaction.response.send_message(
                "Invalid URL.", ephemeral=True
            )

    await interaction.response.send_message("Upscaling...")
    logger.info(f"Upscaling image on message {message_id}...")

    def worker(loop):
        start_time = time.time()

        try:
            upscaled = pipelines.esrgan.upscale(esrgan, image)
        except Exception as e:
            traceback.print_exc()
            return utils.edit(loop, interaction, utils.error(e))

        outpath = os.path.join(config.UPSCALED_DIR, f"{interaction.id}.png")
        upscaled.save(outpath)

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
                "Upscale": esrgan.scale,
                "Size": f"{upscaled.width}x{upscaled.height}",
                "Original Size": f"{image.width}x{image.height}",
            },
        )

        file = discord.File(
            outpath,
            "upscaled.png",
            spoiler=attachment.is_spoiler() if message_id or (url is None) else False,
        )
        utils.edit(loop, interaction, embed=embed, attachments=[file])

        logger.info(f"Finished upscaling image on message {message_id}.")

    thread = Thread(target=worker, args=(asyncio.get_event_loop(),))
    thread.start()
