#!/usr/bin/env python

from __future__ import annotations

import pathlib
import shlex
import subprocess
import tempfile

import gradio as gr
from omegaconf import OmegaConf

from utils import get_timestamp


def gen_feature_extraction_config(exp_name: str, init_image_path: str) -> str:
    config = OmegaConf.load(
        'plug-and-play/configs/pnp/feature-extraction-real.yaml')
    config.config.experiment_name = exp_name
    config.config.init_img = init_image_path
    temp_file = tempfile.NamedTemporaryFile(suffix='.yaml', delete=False)
    with open(temp_file.name, 'w') as f:
        f.write(OmegaConf.to_yaml(config))
    return temp_file.name


def run_feature_extraction_command(init_image_path: str) -> tuple[str, str]:
    exp_name = get_timestamp()
    config_path = gen_feature_extraction_config(exp_name, init_image_path)
    subprocess.run(shlex.split(
        f'python run_features_extraction.py --config {config_path}'),
                   cwd='plug-and-play')
    return f'plug-and-play/experiments/{exp_name}/samples/0.png', exp_name


def gen_pnp_config(
    exp_name: str,
    prompt: str,
    guidance_scale: float,
    ddim_steps: int,
    feature_injection_threshold: int,
    negative_prompt: str,
    negative_prompt_alpha: float,
    negative_prompt_schedule: str,
) -> str:
    config = OmegaConf.load('plug-and-play/configs/pnp/pnp-real.yaml')
    config.source_experiment_name = exp_name
    config.prompts = [prompt]
    config.scale = guidance_scale
    config.num_ddim_sampling_steps = ddim_steps
    config.feature_injection_threshold = feature_injection_threshold
    config.negative_prompt = negative_prompt
    config.negative_prompt_alpha = negative_prompt_alpha
    config.negative_prompt_schedule = negative_prompt_schedule
    temp_file = tempfile.NamedTemporaryFile(suffix='.yaml', delete=False)
    with open(temp_file.name, 'w') as f:
        f.write(OmegaConf.to_yaml(config))
    return temp_file.name


def run_pnp_command(
    exp_name: str,
    prompt: str,
    negative_prompt: str,
    guidance_scale: float,
    ddim_steps: int,
    feature_injection_threshold: int,
    negative_prompt_alpha: float,
    negative_prompt_schedule: str,
) -> str:
    config_path = gen_pnp_config(
        exp_name,
        prompt,
        guidance_scale,
        ddim_steps,
        feature_injection_threshold,
        negative_prompt,
        negative_prompt_alpha,
        negative_prompt_schedule,
    )
    subprocess.run(shlex.split(f'python run_pnp.py --config {config_path}'),
                   cwd='plug-and-play')

    out_dir = pathlib.Path(
        f'plug-and-play/experiments/{exp_name}/translations/{guidance_scale}_{prompt.replace(" ", "_")}'
    )
    out_label = f'INJECTION_T_{feature_injection_threshold}_STEPS_{ddim_steps}_NP-ALPHA_{negative_prompt_alpha}_SCHEDULE_{negative_prompt_schedule}_NP_{negative_prompt.replace(" ", "_")}'
    out_path = out_dir / f'{out_label}_sample_0.png'
    return out_path.as_posix()


def create_real_image_demo():
    with gr.Blocks() as demo:
        with gr.Box():
            gr.Markdown('Step 1')
            with gr.Row():
                with gr.Column():
                    image = gr.Image(label='Input image', type='filepath')
                    extract_feature_button = gr.Button(
                        'Reconstruct and extract features')
                with gr.Column():
                    reconstructed_image = gr.Image(label='Reconstructed image',
                                                   type='filepath')
                    exp_name = gr.Variable()
        with gr.Box():
            gr.Markdown('Step 2')
            with gr.Row():
                with gr.Column():
                    translation_prompt = gr.Text(
                        label='Prompt for translation')
                    negative_prompt = gr.Text(label='Negative prompt')
                    with gr.Accordion(label='Advanced settings', open=False):
                        guidance_scale = gr.Slider(label='Guidance scale',
                                                   minimum=0,
                                                   maximum=50,
                                                   step=0.1,
                                                   value=10)
                        ddim_steps = gr.Slider(
                            label='Number of inference steps',
                            minimum=1,
                            maximum=100,
                            step=1,
                            value=50)
                        feature_injection_threshold = gr.Slider(
                            label='Feature injection threshold',
                            minimum=0,
                            maximum=100,
                            step=1,
                            value=40)
                        negative_prompt_alpha = gr.Slider(
                            label='Negative prompt alpha',
                            minimum=0,
                            maximum=1,
                            step=0.01,
                            value=1)
                        negative_prompt_scheduler = gr.Dropdown(
                            label='Negative prompt schedule',
                            choices=['linear', 'constant', 'exp'],
                            value='linear')
                    generate_button = gr.Button('Generate')
                with gr.Column():
                    result = gr.Image(label='Result', type='filepath')

        with gr.Row():
            gr.Examples(examples=[
                [
                    'plug-and-play/data/horse.png',
                    'a photo of a robot horse',
                    'a photo of a white horse',
                ],
                [
                    'plug-and-play/data/horse.png',
                    'a photo of a bronze horse in a museum',
                    'a photo of a white horse',
                ],
            ],
                        inputs=[
                            image,
                            translation_prompt,
                            negative_prompt,
                        ])

        extract_feature_button.click(
            fn=run_feature_extraction_command,
            inputs=image,
            outputs=[
                reconstructed_image,
                exp_name,
            ],
        )
        generate_button.click(
            fn=run_pnp_command,
            inputs=[
                exp_name,
                translation_prompt,
                negative_prompt,
                guidance_scale,
                ddim_steps,
                feature_injection_threshold,
                negative_prompt_alpha,
                negative_prompt_scheduler,
            ],
            outputs=result,
        )

    return demo


if __name__ == '__main__':
    demo = create_real_image_demo()
    demo.queue().launch()
