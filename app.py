#!/usr/bin/env python

from __future__ import annotations

import os
import pathlib
import shlex
import subprocess

import gradio as gr
import torch

from app_generated_image import create_prompt_demo
from app_real_image import create_real_image_demo

DESCRIPTION = '# [Plug-and-Play diffusion features](https://github.com/MichalGeyer/plug-and-play)'

if (SPACE_ID := os.getenv('SPACE_ID')) is not None:
    DESCRIPTION += f'\n<p>For faster inference without waiting in queue, you may duplicate the space and upgrade to GPU in settings. <a href="https://huggingface.co/spaces/{SPACE_ID}?duplicate=true"><img style="display: inline; margin-top: 0em; margin-bottom: 0em" src="https://bit.ly/3gLdBN6" alt="Duplicate Space" /></a></p>'
if not torch.cuda.is_available():
    DESCRIPTION += '\n<p>Running on CPU 🥶 This demo does not work on CPU.</p>'

if torch.cuda.is_available():
    weight_dir = pathlib.Path('plug-and-play/models/ldm/stable-diffusion-v1')
    if not weight_dir.exists():
        subprocess.run(
            shlex.split(
                'mkdir -p plug-and-play/models/ldm/stable-diffusion-v1/'))
        subprocess.run(
            shlex.split(
                'wget https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4.ckpt -O plug-and-play/models/ldm/stable-diffusion-v1/model.ckpt'
            ))

with gr.Blocks(css='style.css') as demo:
    gr.Markdown(DESCRIPTION)
    with gr.Tabs():
        with gr.TabItem('Use real image as input'):
            create_real_image_demo()
        with gr.TabItem('Use prompt as input'):
            create_prompt_demo()
demo.queue(api_open=False, max_size=10).launch()
