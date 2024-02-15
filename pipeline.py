from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline,
)
import utils


class AllInOnePipeline:
    def __init__(
        self,
        path: str,
        device: str,
        pipe_setup_func,
        embeddings: list[str] = [],
        loras: list[str] = [],
        **kwargs,
    ) -> None:
        factory_func = (
            StableDiffusionPipeline.from_single_file
            if path.endswith(".safetensors")
            else StableDiffusionPipeline.from_pretrained
        )
        self.text2img = factory_func(path, **kwargs).to(device)

        # load embeddings and loras
        for embed in embeddings:
            fn = utils.get_filename(embed)
            self.text2img.load_textual_inversion(embed, token=fn)

        for lora in loras:
            self.text2img.load_lora_weights(lora)

        pipe_setup_func(self.text2img)

        self.img2img = StableDiffusionImg2ImgPipeline(**self.text2img.components).to(
            device
        )
        pipe_setup_func(self.img2img)

        self.inpaint = StableDiffusionInpaintPipeline(**self.text2img.components).to(
            device
        )
        pipe_setup_func(self.inpaint)
