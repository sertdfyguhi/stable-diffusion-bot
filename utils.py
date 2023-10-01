from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from torchvision.transforms import ToTensor
from pipeline import AllInOnePipeline
from dataclasses import dataclass
from termcolor import colored
from torch import Generator
from PIL import Image
import numpy as np
import discord
import asyncio
import torch
import os

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


@dataclass(frozen=True)
class BaseGenerationRequest:
    interaction: discord.Interaction
    model: str
    prompt: str
    negative_prompt: str
    guidance_scale: float
    step_count: int
    seed: int


@dataclass(frozen=True)
class Text2ImgGenerationRequest(BaseGenerationRequest):
    width: int
    height: int

    ptype = "text2img"


@dataclass(frozen=True)
class Img2ImgGenerationRequest(BaseGenerationRequest):
    image: Image.Image

    ptype = "img2img"


@dataclass(frozen=True)
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


def bold(msg: str) -> str:
    return colored(msg, attrs=["bold"])


def edit(
    loop: asyncio.AbstractEventLoop,
    interaction: discord.Interaction,
    msg: str = None,
    embed: discord.Embed = None,
    attachments: [discord.File] = [],
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
        custom_pipeline="lpw_stable_diffusion",
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
