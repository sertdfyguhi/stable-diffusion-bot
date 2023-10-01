TOKEN = ""
SAVE_DIR = "saves"

# name and path to models
# example:
# {
#     "aurora": "models/aurora-v2",
#     "hassaku": "models/hassaku.safetensors"
# }
MODEL_PATHS = {}

DEVICE = "cuda"

# list of paths to embedding and lora models
EMBEDDINGS = []
LORAS = []

PROGRESS_EMBED_COLOR = "blue"
RESULT_EMBED_COLOR = "random"

MAX_WIDTH = 800
MAX_HEIGHT = 1024


from diffusers import DPMSolverMultistepScheduler


def pipeline_setup(pipeline):
    # sets pipeline scheduler
    # if changed be sure to change scheduler name at line 154 & 179 in main.py
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
        pipeline.scheduler.config,
        use_karras_sigmas=True,
    )

    # attention slicing tends to improve performance
    # on apple silicon laptops with ram over 64gb
    # DO NOT REMOVE unless you have >=64gb of ram
    if DEVICE == "mps":
        pipeline.enable_attention_slicing()
