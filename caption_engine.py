"""
caption_engine.py
------------------
Wraps the HuggingFace BLIP image-captioning model and exposes:
  - CaptionEngine: loads the model once (cached by Streamlit) and generates captions
  - STYLE_PROMPTS: the "style" trick — BLIP supports *conditional* captioning,
    where you feed it a text prefix and it completes the sentence. Different
    prefixes nudge the tone of the output. We also apply light template
    post-processing for styles that need more than a prefix can give (e.g. emojis).
"""

from dataclasses import dataclass
from typing import Optional

import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration




def _social_media(caption: str) -> str:
    return f"{caption.capitalize()} ✨📸 #photography #moment"


def _seo(caption: str) -> str:
    # crude keyword-forward rewrite: strip filler words, title-case key nouns
    return caption.capitalize() + "."


def _dramatic(caption: str) -> str:
    return f"Behold — {caption}."


STYLE_PROMPTS = {
    "Descriptive (default)": {"prompt": None, "postprocess": None},
    "Detailed": {"prompt": "a detailed photography of", "postprocess": None},
    "Concise": {"prompt": "a photo of", "postprocess": None},
    "Creative / Artistic": {"prompt": "an artistic depiction of", "postprocess": _dramatic},
    "Social Media": {"prompt": None, "postprocess": _social_media},
    "SEO / Alt-text": {"prompt": "an image showing", "postprocess": _seo},
}


@dataclass
class CaptionResult:
    style: str
    caption: str


class CaptionEngine:
    """Loads BLIP once and reuses it for every caption request."""

    def __init__(self, model_name: str = "Salesforce/blip-image-captioning-base"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def _raw_caption(self, image: Image.Image, prompt: Optional[str], max_length: int, num_beams: int) -> str:
        if prompt:
            inputs = self.processor(image, prompt, return_tensors="pt").to(self.device)
        else:
            inputs = self.processor(image, return_tensors="pt").to(self.device)

        out_ids = self.model.generate(
            **inputs,
            max_length=max_length,
            num_beams=num_beams,
        )
        text = self.processor.decode(out_ids[0], skip_special_tokens=True)

        # BLIP echoes the prompt back at the start of conditional generations;
        # strip it so we don't get "a detailed photography of a detailed photography of a cat"
        if prompt and text.lower().startswith(prompt.lower()):
            text = text[len(prompt):].strip()
        return text

    def generate_one_style(
        self,
        image: Image.Image,
        style: str,
        max_length: int = 50,
        num_beams: int = 5,
    ) -> CaptionResult:
        config = STYLE_PROMPTS[style]
        caption = self._raw_caption(image, config["prompt"], max_length, num_beams)
        if config["postprocess"]:
            caption = config["postprocess"](caption)
        return CaptionResult(style=style, caption=caption)

    def generate_many_styles(
        self,
        image: Image.Image,
        styles: list[str],
        max_length: int = 50,
        num_beams: int = 5,
    ) -> list[CaptionResult]:
        return [self.generate_one_style(image, s, max_length, num_beams) for s in styles]
