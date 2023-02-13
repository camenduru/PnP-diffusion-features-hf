#!/usr/bin/env python

from __future__ import annotations

import pathlib
import shlex
import subprocess
import tempfile

import gradio as gr
from omegaconf import OmegaConf


def gen_feature_extraction_config(
    exp_name: str,
    prompt: str,
    seed: int,
    guidance_scale: float,
    ddim_steps: int,
) -> str:
    config = OmegaConf.load(
        'plug-and-play/configs/pnp/feature-extraction-generated.yaml')
    config.config.experiment_name = exp_name
    config.config.prompt = prompt
    config.config.seed = seed
    config.config.scale = guidance_scale
    config.config.ddim_steps = ddim_steps
    temp_file = tempfile.NamedTemporaryFile(suffix='.yaml', delete=False)
    with open(temp_file.name, 'w') as f:
        f.write(OmegaConf.to_yaml(config))
    return temp_file.name


def run_feature_extraction_command(
    prompt: str,
    seed: int,
    guidance_scale: float,
    ddim_steps: int,
) -> tuple[str, str]:
    exp_name = f'{prompt.replace(" ", "_")}_{seed}_{guidance_scale:.1f}_{ddim_steps}'
    if not pathlib.Path(f'plug-and-play/experiments/{exp_name}').exists():
        config_path = gen_feature_extraction_config(
            exp_name,
            prompt,
            seed,
            guidance_scale,
            ddim_steps,
        )
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
    config = OmegaConf.load('plug-and-play/configs/pnp/pnp-generated.yaml')
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


def process_example(source_prompt: str, seed: int,
                    translation_prompt: str) -> tuple[str, str, str]:
    generated_image, exp_name = run_feature_extraction_command(
        source_prompt, seed, guidance_scale=5, ddim_steps=50)
    result = run_pnp_command(exp_name,
                             translation_prompt,
                             negative_prompt='',
                             guidance_scale=7.5,
                             ddim_steps=50,
                             feature_injection_threshold=40,
                             negative_prompt_alpha=0.75,
                             negative_prompt_schedule='linear')
    return generated_image, exp_name, result


def create_prompt_demo() -> gr.Blocks:
    with gr.Blocks() as demo:
        with gr.Box():
            gr.Markdown(
                'Step 1 (This step will take about 1.5 minutes on A10G.)')
            with gr.Row():
                with gr.Column():
                    source_prompt = gr.Text(label='Source prompt')
                    seed = gr.Slider(label='Seed',
                                     minimum=0,
                                     maximum=100000,
                                     step=1,
                                     value=0)
                    with gr.Accordion(label='Advanced settings', open=False):
                        source_guidance_scale = gr.Slider(
                            label='Guidance scale',
                            minimum=0,
                            maximum=50,
                            step=0.1,
                            value=5)
                        source_ddim_steps = gr.Slider(label='DDIM steps',
                                                      minimum=1,
                                                      maximum=100,
                                                      step=1,
                                                      value=50)
                    extract_feature_button = gr.Button(
                        'Generate and extract features')
                with gr.Column():
                    generated_image = gr.Image(label='Generated image',
                                               type='filepath')
                    exp_name = gr.Text(visible=False)
        with gr.Box():
            gr.Markdown(
                'Step 2 (This step will take about 1.5 minutes on A10G.)')
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
                                                   value=7.5)
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
                            value=0.75)
                        negative_prompt_schedule = gr.Dropdown(
                            label='Negative prompt schedule',
                            choices=['linear', 'constant', 'exp'],
                            value='linear')
                    generate_button = gr.Button('Generate')
                with gr.Column():
                    result = gr.Image(label='Result', type='filepath')
        with gr.Row():
            gr.Examples(
                examples=[
                    ['horse in mud', 50, 'a photo of a zebra in the snow'],
                    ['horse in mud', 50, 'a photo of a husky in the grass'],
                ],
                inputs=[
                    source_prompt,
                    seed,
                    translation_prompt,
                ],
                outputs=[
                    generated_image,
                    exp_name,
                    result,
                ],
                fn=process_example,
                cache_examples=True,
            )

        extract_feature_button.click(
            fn=run_feature_extraction_command,
            inputs=[
                source_prompt,
                seed,
                source_guidance_scale,
                source_ddim_steps,
            ],
            outputs=[
                generated_image,
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
                negative_prompt_schedule,
            ],
            outputs=result,
        )
    return demo


if __name__ == '__main__':
    demo = create_prompt_demo()
    demo.queue().launch()
