# Stable Diffusion Discord Bot

A discord bot for stable diffusion image generation using diffusers.

# Commands

- /generate: Generates an image using a prompt.
- /stop: Stops current generation.
- /stop_all: Stops all generations requested by user.

# Setup

1. Clone this repository.

```sh
git clone https://github.com/sertdfyguhi/stable-diffusion-bot.git
```

2. Install required Python packages.

```sh
pip3 install -r requirements.txt
```

3. Update `config.py` with bot token and model paths.
4. Finally, run `main.py`!

```sh
py main.py
```

# Todo

- [x] Img2img generation
- [ ] Resizing input image in img2img
- [ ] Upscaling image
- [ ] Inpaint generation
