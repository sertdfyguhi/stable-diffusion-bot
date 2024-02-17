import discord
import config

NAME = "models"
DESCRIPTION = "Lists all available models."


async def command(interaction: discord.Interaction):
    embed = discord.Embed(
        title="Models Available",
        description="- " + "\n- ".join(config.MODEL_PATHS.keys()),
        color=config.PRIMARY_EMBED_COLOR,
    )

    await interaction.response.send_message(embed=embed)
