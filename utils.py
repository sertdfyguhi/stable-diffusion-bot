from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from torchvision.transforms import ToTensor, ToPILImage
from dataclasses import dataclass
from termcolor import colored
from torch import Generator
import numpy as np
import discord
import asyncio
import torch
import os

to_tensor = ToTensor()
to_pil = ToPILImage()

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
class GenerationRequest:
    interaction: discord.Interaction
    model: str
    prompt: str
    negative_prompt: str
    guidance_scale: float
    step_count: int
    seed: int
    width: int
    height: int


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
    embeddings: list[str] = [],
    loras: list[str] = [],
):
    factory_func = (
        StableDiffusionPipeline.from_single_file
        if model_path.endswith(".safetensors")
        else StableDiffusionPipeline.from_pretrained
    )
    pipeline = factory_func(
        model_path,
        custom_pipeline="lpw_stable_diffusion",
        # torch_dtype=torch.float16,
    )
    pipeline = pipeline.to(device)

    if device == "mps":
        pipeline.enable_attention_slicing()

    # disable safety checker
    pipeline.orig_safety_checker = pipeline.safety_checker
    pipeline.safety_checker = None

    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
        pipeline.scheduler.config,
        use_karras_sigmas=True,
    )

    for embed in embeddings:
        fn = os.path.basename(embed).split(".")[0]
        pipeline.load_textual_inversion(embed, token=fn)

    for lora in loras:
        pipeline.load_lora_weights(lora)

    return pipeline


def load_models(
    models: dict[str, str],
    device: str,
    embeddings: list[str] = [],
    loras: list[str] = [],
) -> dict[str, StableDiffusionPipeline]:
    models = models.copy()

    for key, path in models.items():
        models[key] = load_model(path, device, embeddings, loras)

    return models


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
    return to_pil((255 * image).numpy().astype(np.uint8))


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
