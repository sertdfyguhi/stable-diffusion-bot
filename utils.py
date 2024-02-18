# https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/14186/
# Hack to fix a changed import in torchvision 0.17+, which otherwise breaks
# basicsr; see https://github.com/AUTOMATIC1111/stable-diffusion-webui/issues/13985
import sys

try:
    import torchvision.transforms.functional_tensor  # type: ignore
except ImportError:
    try:
        import torchvision.transforms.functional as functional

        sys.modules["torchvision.transforms.functional_tensor"] = functional
    except ImportError:
        pass  # shrug...

from diffusers.utils.loading_utils import load_image
from basicsr.archs.rrdbnet_arch import RRDBNet
from PIL import Image, UnidentifiedImageError
from diffusers import StableDiffusionPipeline
from torchvision.transforms import ToTensor
from pipeline import AllInOnePipeline
from realesrgan import RealESRGANer
from dataclasses import dataclass
from torch import Generator
import numpy as np
import discord
import asyncio
import torch
import os
import io

to_tensor = ToTensor()

# factors for fast latent decoding
rgb_latent_factors = torch.Tensor(
    [
        [0.298, 0.207, 0.208],
        [0.187, 0.286, 0.173],
        [-0.158, 0.189, 0.264],
        [-0.184, -0.271, -0.473],
    ]
)


@dataclass
class BaseGenerationRequest:
    interaction: discord.Interaction
    model: str
    prompt: str
    negative_prompt: str | None
    guidance_scale: float
    step_count: int
    seed: int


@dataclass
class Text2ImgGenerationRequest(BaseGenerationRequest):
    width: int
    height: int

    ptype = "text2img"


@dataclass
class Img2ImgGenerationRequest(BaseGenerationRequest):
    image: Image.Image

    ptype = "img2img"


@dataclass
class InpaintGenerationRequest(BaseGenerationRequest):
    mask: Image.Image

    ptype = "inpaint"


GenerationRequest = (
    BaseGenerationRequest
    | Text2ImgGenerationRequest
    | Img2ImgGenerationRequest
    | InpaintGenerationRequest
)


def error(msg):
    return f"**Error:** {msg}"


def edit(
    loop: asyncio.AbstractEventLoop,
    interaction: discord.Interaction,
    msg: str = None,
    embed: discord.Embed = None,
    attachments: list[discord.File] = [],
):
    asyncio.run_coroutine_threadsafe(
        interaction.edit_original_response(
            content=msg,
            embed=embed,
            attachments=attachments,
        ),
        loop,
    )


def load_model(
    model_path: str,
    device: str,
    pipe_setup_func,
    embeddings: list[str] = [],
    loras: list[str] = [],
):
    return AllInOnePipeline(
        model_path,
        device,
        pipe_setup_func,
        embeddings,
        loras,
    )


def load_models(
    models: dict[str, str],
    device: str,
    pipe_setup_func,
    embeddings: list[str] = [],
    loras: list[str] = [],
) -> dict[str, AllInOnePipeline]:
    return {
        key: load_model(path, device, pipe_setup_func, embeddings, loras)
        for key, path in models.items()
    }


def load_esrgan_model(
    model_path: str,
    device: str,
    model_name: str = None,
):
    if model_name is None:
        model_name = get_filename(model_path)

    if model_name in [
        "RealESRGAN_x4plus",
        "RealESRNet_x4plus",
    ]:  # x4 RRDBNet model
        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=4,
        )
    elif model_name == "RealESRGAN_x4plus_anime_6B":  # x4 RRDBNet model with 6 blocks
        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=6,
            num_grow_ch=32,
            scale=4,
        )
    elif model_name == "RealESRGAN_x2plus":  # x2 RRDBNet model
        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=2,
        )
    else:
        raise ValueError("Failed to determine ESRGAN model type.")

    esrgan_model = RealESRGANer(
        scale=model.scale,
        device=device,
        model_path=model_path,
        model=model,
    )
    esrgan_model.model_name = model_name

    return esrgan_model


def create_torch_generator(seed: int | None = None, device: str = "cpu") -> Generator:
    gen = Generator(device=device)

    if seed:
        gen.manual_seed(seed)
    else:
        gen.seed()

    return gen


def path_join(paths: list[str], sep: str = ";") -> str:
    return sep.join([path.replace(",", "\\,").replace(";", "\\;") for path in paths])


def fast_decode(latent: torch.Tensor):
    image = latent.permute(1, 2, 0).cpu() @ rgb_latent_factors
    return Image.fromarray((255 * image).numpy().astype(np.uint8))


def check_img_nsfw(pipeline: StableDiffusionPipeline, image) -> bool | None:
    if (
        not hasattr(pipeline, "orig_safety_checker")
        or pipeline.orig_safety_checker is None
    ):
        return None

    features = pipeline.feature_extractor([image], return_tensors="pt")
    features = features.to(pipeline._execution_device)

    _, has_nsfw_concept = pipeline.orig_safety_checker(
        images=to_tensor(image),
        clip_input=features.pixel_values.to(pipeline.text_encoder.dtype),
    )

    return has_nsfw_concept[0]


def add_fields(embed: discord.Embed, fields: dict) -> None:
    for name, value in fields.items():
        embed.add_field(name=name, value=value)


def get_embed_color(color: str | list | tuple) -> discord.Colour:
    if type(color) in [list, tuple]:
        return discord.Color.from_rgb(color)
    elif hasattr(discord.Color, color):
        return getattr(discord.Color, color)()
    else:
        return discord.Color.from_str(color)


def get_filename(path: str) -> str:
    return os.path.basename(path).split(".")[0]


async def load_image_from_message(
    message: discord.Message,
) -> tuple[Image.Image, discord.Attachment]:
    attachments = message.attachments
    if len(attachments) == 0:
        raise ValueError(f"There are no attachments attached to the message.")

    try:
        attachment = next(
            attachment
            for attachment in attachments
            if attachment.content_type.startswith("image")
        )
    except StopIteration:
        raise ValueError(f"There are no images attached to the message.")

    try:
        # open file bytes as PIL image
        image = Image.open(io.BytesIO(await attachment.read()))
    except UnidentifiedImageError:
        raise ValueError("File could not be read as an image.")

    return image, attachment


async def load_image_from_args(
    message_id: str = None,
    url: str = None,
    file: discord.Attachment = None,
    channel: discord.TextChannel = None,
) -> tuple[Image.Image, None | discord.Attachment]:
    attachment = None

    if message_id:
        try:
            message = await channel.fetch_message(int(message_id))
        except ValueError:
            raise ValueError("Message ID is not an ID.")
        except Exception:
            raise ValueError(f"Could not find message {message_id}!")

        image, attachment = await load_image_from_message(message)
    elif url:
        try:
            image = load_image(url)
        except UnidentifiedImageError:
            raise ValueError("URL response could not be read as an image.")
        except ValueError:
            raise ValueError("Invalid URL.")
    elif file:
        if not file.content_type.startswith("image/"):
            raise ValueError("File provided is not an image.")

        try:
            # open file bytes as PIL image
            image = Image.open(io.BytesIO(await file.read()))
        except UnidentifiedImageError:
            raise ValueError("File could not be read as an image.")
    else:
        raise ValueError("No image provided.")

    return image, attachment
