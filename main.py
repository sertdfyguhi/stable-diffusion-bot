from PIL.PngImagePlugin import PngInfo
from threading import Thread
import pipelines
import traceback
import discord
import asyncio
import inspect
import logging
import config
import utils
import time
import io
import os


# logging setup
format_str = f"[%(asctime)s] {utils.bold('%(levelname)s')} :: {utils.bold('%(name)s')} - %(message)s"
logging.basicConfig(
    level=logging.DEBUG,
    format=format_str,
    datefmt="%d/%m/%Y %H:%M:%S",
)
logger = logging.getLogger(__name__)

# load config embed colors
primary_embed_color = utils.get_embed_color(config.PRIMARY_EMBED_COLOR)
secondary_embed_color = utils.get_embed_color(config.SECONDARY_EMBED_COLOR)

# client setup
intents = discord.Intents.default()
intents.message_content = True

client = discord.Client(intents=intents)
tree = discord.app_commands.CommandTree(client)

logger.info("Loading models...")

# load models into stable diffusion pipeline
models = utils.load_models(
    config.MODEL_PATHS,
    device=config.DEVICE,
    pipe_setup_func=config.pipeline_setup,
    embeddings=config.EMBEDDINGS,
    loras=config.LORAS,
)

logger.info("Finished loading models.")


current_user_id = None  # current generating user id
stopped = False
queue: list[utils.GenerationRequest] = []


class GenerationExit(Exception):
    pass


def generate(loop):
    global current_user_id

    while len(queue) > 0:
        req = queue.pop()
        interaction = req.interaction

        current_user_id = interaction.user.id

        last_step_time = time.time()
        start_time = last_step_time

        # generation callback to update interaction message
        def callback(step, _, latents):
            global stopped
            nonlocal last_step_time

            now = time.time()
            eta = (now - start_time) / (step + 1) * (req.step_count - step)

            embed = discord.Embed(
                title=f"Generating... ({step}/{req.step_count})",
                color=secondary_embed_color,
            )
            embed.set_image(url="attachment://preview.png")
            embed.add_field(name="Step Time", value=f"{now - last_step_time:.2f}s")
            embed.add_field(name="ETA", value=f"{eta:.2f}s")

            # use bytesio to avoid saving image to disk
            with io.BytesIO() as img_bin:
                preview = utils.fast_decode(latents[0])
                preview.save(img_bin, "png")
                img_bin.seek(0)

                utils.edit(
                    loop,
                    interaction,
                    embed=embed,
                    attachments=[discord.File(img_bin, "preview.png")],
                )

            last_step_time = time.time()

            # check if generation has been stopped
            if stopped:
                stopped = False
                raise GenerationExit()

        generator = utils.create_torch_generator(req.seed)
        pipeline = getattr(models[req.model], req.ptype)
        base_kwargs = {
            "prompt": req.prompt,
            "negative_prompt": req.negative_prompt,
            "guidance_scale": req.guidance_scale,
            "num_inference_steps": req.step_count,
            "generator": generator,
            "callback": callback,
        }

        try:
            match req.ptype:
                case "text2img":
                    image = pipeline(
                        **base_kwargs,
                        width=req.width,
                        height=req.height,
                    ).images[0]

                case "img2img":
                    image = pipeline(**base_kwargs, image=req.image).images[0]

                case "inpaint":
                    image = pipeline(**base_kwargs, image=req.mask).images[0]
        except GenerationExit:
            continue
        except Exception as e:
            traceback.print_exc()
            utils.edit(loop, interaction, utils.error(e))
            continue

        is_nsfw_image = utils.check_img_nsfw(pipeline, image)

        gen_seed = generator.initial_seed()

        # save generated image
        pnginfo = PngInfo()
        pnginfo.add_text("prompt", req.prompt)
        pnginfo.add_text("negative_prompt", req.negative_prompt)
        pnginfo.add_text("guidance_scale", str(req.guidance_scale))
        pnginfo.add_text("step_count", str(req.step_count))
        pnginfo.add_text("seed", str(gen_seed))
        pnginfo.add_text("model", config.MODEL_PATHS[req.model])
        pnginfo.add_text("pipeline", req.ptype)
        pnginfo.add_text("scheduler", "DPMSolverMultistepScheduler Karras++")
        pnginfo.add_text("embeddings", utils.path_join(config.EMBEDDINGS))
        pnginfo.add_text("loras", utils.path_join(config.LORAS))

        filename = f"{req.interaction.id}.png"
        out_path = os.path.join(config.SAVE_DIR, filename)
        image.save(out_path, pnginfo=pnginfo)

        # create embed
        embed = discord.Embed(
            title=f"Generated Image (Total Time: {time.time() - start_time:.2f}s)",
            color=primary_embed_color,
        )

        author = req.interaction.user
        embed.set_author(name=author.display_name, icon_url=author.avatar.url)
        embed.set_image(url=f"attachment://{filename}")
        utils.add_fields(
            embed,
            {
                "Model": req.model,
                "Prompt": req.prompt,
                "Negative Prompt": req.negative_prompt,
                "Guidance / CFG Scale": req.guidance_scale,
                "Step Count": req.step_count,
                "Scheduler / Sampler": "DPMSolverMultistepScheduler Karras++",
                "Seed": gen_seed,
                "Size": f"{image.width}x{image.height}",
            },
        )

        # create file attachment
        file = discord.File(
            out_path,
            filename,
            spoiler=True if is_nsfw_image is None else is_nsfw_image,
        )

        utils.edit(
            loop,
            interaction,
            embed=embed,
            attachments=[file],
        )

        logger.info(f"Finished generation for {req.interaction.id}.")

    logger.info("Finished generating queue.")
    current_user_id = None


async def stop_current_gen():
    global stopped
    stopped = True

    # wait until generation stops
    while stopped:
        await asyncio.sleep(0.1)


# create discord commands
for module in pipelines.commands:
    # wrapper avoids module being overwritten by subsequent iterations
    def wrapper(module):
        async def command(interaction: discord.Interaction, *args, **kwargs):
            global current_user_id

            try:
                res = module.handle(interaction, *args, **kwargs)
            except Exception as e:
                return await interaction.response.send_message(utils.error(e))

            # validate parameters
            if res.model not in models:
                return await interaction.response.send_message(
                    utils.error(
                        f'No model named "{res.model}". List all available models using /models.'
                    )
                )

            queue.append(res)

            if current_user_id:
                return await interaction.response.send_message(
                    f"Currently generating, you are {len(queue) + 1} in the queue."
                )

            await interaction.response.send_message("Starting generation...")
            logger.info("Starting generation...")

            current_user_id = interaction.user.id
            thread = Thread(target=generate, args=(asyncio.get_event_loop(),))
            thread.start()

        return command

    command = wrapper(module)

    # update function signature to have same params as handle functions
    command.__signature__ = inspect.signature(module.handle)
    tree.command(name=module.NAME, description=module.DESCRIPTION)(command)


@tree.command(name="stop", description="Stops generation.")
async def stop_cmd(interaction: discord.Interaction):
    # check if interaction author is the same as generation requester
    if current_user_id is None or current_user_id != interaction.user.id:
        return await interaction.response.send_message(
            utils.error("You are not generating anything."), ephemeral=True
        )
    else:
        await interaction.response.send_message("Stopping...")
        await stop_current_gen()
        await interaction.edit_original_response(content="Stopped.")


@tree.command(name="stop_all", description="Stops all generation requested by user.")
async def stop_all_cmd(interaction: discord.Interaction):
    await interaction.response.send_message("Stopping all generations...")
    requester = interaction.user.id

    for i, req in enumerate(queue):
        if req.interaction.user.id == requester:
            del queue[i]

    if current_user_id == requester:
        await stop_current_gen()

    await interaction.edit_original_response(content="Stopped.")


@tree.command(name="models", description="Lists all available models.")
async def models_cmd(interaction: discord.Interaction):
    embed = discord.Embed(
        title="Models Available",
        description="- " + "\n- ".join(models.keys()),
        color=primary_embed_color,
    )

    await interaction.response.send_message(embed=embed)


@client.event
async def on_ready():
    await tree.sync()
    logger.info(f"Logged in as {client.user}.")


if __name__ == "__main__":
    client.run(config.TOKEN)
