import os
os.environ["NEURON_FUSE_SOFTMAX"] = "1"

import torch
import torch.nn as nn
import torch_neuronx
import numpy as np

from matplotlib import pyplot as plt
from matplotlib import image as mpimg
import time
import copy
from IPython.display import clear_output

from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.models.unet_2d_condition import UNet2DConditionOutput

# Compatibility for diffusers<0.18.0
from packaging import version
import diffusers
diffusers_version = version.parse(diffusers.__version__)
use_new_diffusers = diffusers_version >= version.parse('0.18.0')
if use_new_diffusers:
    from diffusers.models.attention_processor import Attention
else:
    from diffusers.models.cross_attention import CrossAttention
import gradio as gr
# Define datatype
DTYPE = torch.float32


class UNetWrap(nn.Module):
    def __init__(self, unet):
        super().__init__()
        self.unet = unet
 
    def forward(self, sample, timestep, encoder_hidden_states, text_embeds=None, time_ids=None):
        out_tuple = self.unet(sample,
                              timestep,
                              encoder_hidden_states,
                              added_cond_kwargs={"text_embeds": text_embeds, "time_ids": time_ids},
                              return_dict=False)
        return out_tuple
    
    
class NeuronUNet(nn.Module):
    def __init__(self, unetwrap):
        super().__init__()
        self.unetwrap = unetwrap
        self.config = unetwrap.unet.config
        self.in_channels = unetwrap.unet.in_channels
        self.add_embedding = unetwrap.unet.add_embedding
        self.device = unetwrap.unet.device
 
    def forward(self, sample, timestep, encoder_hidden_states, added_cond_kwargs=None, return_dict=False, cross_attention_kwargs=None):
        sample = self.unetwrap(sample,
                               timestep.to(dtype=DTYPE).expand((sample.shape[0],)),
                               encoder_hidden_states,
                               added_cond_kwargs["text_embeds"],
                               added_cond_kwargs["time_ids"])[0]
        return UNet2DConditionOutput(sample=sample)
    
    
# --- Load all compiled models and run pipeline ---
COMPILER_WORKDIR_ROOT = 'sdxl_base_compile_dir_1024'
model_id = "stabilityai/stable-diffusion-xl-base-1.0"
decoder_filename = os.path.join(COMPILER_WORKDIR_ROOT, 'vae_decoder/model.pt')
unet_filename = os.path.join(COMPILER_WORKDIR_ROOT, 'unet/model.pt')
post_quant_conv_filename = os.path.join(COMPILER_WORKDIR_ROOT, 'vae_post_quant_conv/model.pt')

pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=DTYPE)

# pipe.scheduler.config['num_train_timesteps'] = 30
print('##### PIPE CONFIG: ', pipe.scheduler.config)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)


# Load the compiled UNet onto two neuron cores.
pipe.unet = NeuronUNet(UNetWrap(pipe.unet))
device_ids = [0,1]
pipe.unet.unetwrap = torch_neuronx.DataParallel(torch.jit.load(unet_filename), device_ids, set_dynamic_batching=False)

# Load other compiled models onto a single neuron core.
pipe.vae.decoder = torch.jit.load(decoder_filename)
pipe.vae.post_quant_conv = torch.jit.load(post_quant_conv_filename)

# Run pipeline and save result image
prompt = "a photo of a porsche parked by the street during fall season"
start_time = time.time()
image = pipe(prompt).images[0]
end_time = time.time()
print("warm up latency is: " + str(end_time-start_time) + " seconds")

def text2img(PROMPT, INFERENCE_STEPS=50, GUIDANCE_SCALE=7.5):
    start_time = time.time()
    image = pipe(PROMPT, num_inference_steps=round(INFERENCE_STEPS), guidance_scale=GUIDANCE_SCALE ).images[0]
    end_time = time.time()
    time_elapsed_str = "Elapsed time is: " + str(end_time-start_time) + " seconds"
    inference_steps_str = "Inference steps is " + str(round(INFERENCE_STEPS))
    guidance_scale_str = "Guidance scale is " + str(GUIDANCE_SCALE)
    file_name = "_".join(prompt.split())
    FILE_NAME = file_name + '.png'
    image.save(FILE_NAME)
    return image, time_elapsed_str, inference_steps_str, guidance_scale_str

app = gr.Interface(fn=text2img,
    inputs=["text", gr.Slider(1, 100, step=1, label='Inference steps (bigger is better in quality but slower)', value = 50), 
            gr.Slider(1, 10, label='Guidance scale (bigger means image resembles closer to prompt)', value=7.5)],
    outputs = [gr.Image(height=768, width=768), "text", "text", "text"],
    title = 'Stable Diffusion XL in AWS EC2 Inf2 instance')

app.queue()
app.launch(share = True, debug = False)