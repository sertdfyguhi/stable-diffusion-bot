# WORK IN PROGRESS

import discord


class ToolbarView(discord.ui.View):
    async def on_timeout(self) -> None:
        for item in self.children:
            item.disabled = True

        await self.message.edit(view=self)

    @discord.ui.button(label="Upscale")
    async def upscale_btn(
        self, interaction: discord.Interaction, button: discord.ui.Button
    ):
        print(interaction.message.attachments)
        await interaction.response.send_message("no pussy")
