#!/usr/bin/env python

from __future__ import annotations

import pathlib
import shlex
import subprocess

import gradio as gr

from app_generated_image import create_prompt_demo
from app_real_image import create_real_image_demo

DESCRIPTION = '''# Plug-and-Play diffusion features

This is an unofficial demo for [https://github.com/MichalGeyer/plug-and-play](https://github.com/MichalGeyer/plug-and-play).
'''

weight_dir = pathlib.Path('plug-and-play/models/ldm/stable-diffusion-v1')
if not weight_dir.exists():
    subprocess.run(
        shlex.split('mkdir -p plug-and-play/models/ldm/stable-diffusion-v1/'))
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

demo.queue().launch()
