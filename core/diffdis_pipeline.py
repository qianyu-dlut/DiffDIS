
from typing import Any, Dict, Union
import random
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm
from PIL import Image
from diffusers import (
    DiffusionPipeline,
    DDPMScheduler,
    UNet2DConditionModel,
    AutoencoderKL,
)
from transformers import CLIPTextModel, CLIPTokenizer
from utils.depth_ensemble import ensemble
import torch.nn.functional as F


class DiffDISPipeline(DiffusionPipeline):
    # two hyper-parameters
    rgb_latent_scale_factor = 0.18215
    mask_latent_scale_factor = 0.18215
    weight_dtype = torch.float32
    
    def __init__(self,
                 unet:UNet2DConditionModel,
                 vae:AutoencoderKL,
                 scheduler:DDPMScheduler,
                 text_encoder:CLIPTextModel,
                 tokenizer:CLIPTokenizer,
                 ):
        super().__init__()
            
        self.register_modules(
            unet=unet,
            vae=vae,
            scheduler=scheduler,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
        )
        self.empty_text_embed = None
    
    @torch.no_grad()
    def __call__(self,
                 input_image:Image,
                 denosing_steps: int =10,
                 ensemble_size: int =10,
                 processing_res: int = 1024,
                 match_input_res:bool =True,
                 batch_size:int =0,
                 show_progress_bar:bool = True,
                 ensemble_kwargs: Dict = None,
                 ) -> torch.Tensor:
        # inherit from thea Diffusion Pipeline
        # adjust the input resolution.
        if not match_input_res:
            assert (
                processing_res is not None                
            )," Value Error: `resize_output_back` is only valid with "
        
        rgb_norm = input_image
        duplicated_rgb = torch.stack([rgb_norm] * ensemble_size)
        single_rgb_dataset = TensorDataset(duplicated_rgb)
        
        # find the batch size
        if batch_size>0:
            _bs = batch_size
        else:
            _bs = 1
        
        single_rgb_loader = DataLoader(single_rgb_dataset,batch_size=_bs,shuffle=False)
        mask_pred_ls = []
        edge_pred_ls = []
        
        if show_progress_bar:
            iterable_bar = tqdm(
                single_rgb_loader, desc=" " * 2 + "Inference batches", leave=False
            )
        else:
            iterable_bar = single_rgb_loader
        
        for batch in iterable_bar:
            (batched_image,)= batch  # here the image is around [-1,1]
            mask_pred, edge_pred = self.single_infer(
                input_rgb=batched_image.squeeze(0),
                num_inference_steps=denosing_steps,
                show_pbar=show_progress_bar
            )
            mask_pred_ls.append(mask_pred.detach().clone())
            edge_pred_ls.append(edge_pred.detach().clone())
        
        mask_preds = torch.concat(mask_pred_ls, axis=0).squeeze() 
        edge_preds = torch.concat(edge_pred_ls, axis=0).squeeze() 
        torch.cuda.empty_cache()  # clear vram cache for ensembling
        
        # ----------------- Test-time ensembling -----------------
        if ensemble_size > 1:
            mask_pred, _ = ensemble(mask_preds, **(ensemble_kwargs or {}))
            edge_pred, _ = ensemble(edge_preds, **(ensemble_kwargs or {}))
        else:
            mask_pred = mask_preds
            edge_pred = edge_preds
        
        # ----------------- Post processing -----------------
        # scale prediction to [0, 1]
        mask_pred = (mask_pred - torch.min(mask_pred)) / (torch.max(mask_pred) - torch.min(mask_pred))
        edge_pred = (edge_pred - torch.min(edge_pred)) / (torch.max(edge_pred) - torch.min(edge_pred))
        
        return mask_pred, edge_pred
        
    
    def __encode_empty_text(self):
        """
        Encode text embedding for empty prompt
        """

        prompt = ""
        text_inputs = self.tokenizer(
            prompt,
            padding="do_not_pad",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids.to(self.text_encoder.device) #[1,2]
        self.empty_text_embed = self.text_encoder(text_input_ids)[0].to(self.dtype) #[1,2,1024]


        
    @torch.no_grad()
    def single_infer(self,input_rgb:torch.Tensor,
                     num_inference_steps:int,
                     show_pbar:bool):
        
        
        device = input_rgb.device
        bsz = input_rgb.shape[0]
        
        # set timesteps: inherit from the diffuison pipeline
        self.scheduler.set_timesteps(num_inference_steps, device=device) # here the numbers of the steps is only 10.
        self.scheduler.alphas_cumprod = self.scheduler.alphas_cumprod.cuda()
        timesteps = self.scheduler.timesteps  # [T]
        
        # encode image
        rgb_latent = self.encode_RGB(input_rgb) # 1/8 Resolution with a channel nums of 4. 
        mask_edge_latent = torch.randn(rgb_latent.shape, device=device, dtype=self.dtype).repeat(2,1,1,1) 


        rgb_latent = rgb_latent.to(self.weight_dtype).repeat(2,1,1,1) 
        rgb_resized2_latents = self.encode_RGB(F.interpolate(input_rgb, size=input_rgb.shape[-1]//2, mode='bilinear', align_corners=False)).to(self.weight_dtype).repeat(2,1,1,1)
        rgb_resized4_latents = self.encode_RGB(F.interpolate(input_rgb, size=input_rgb.shape[-1]//4, mode='bilinear', align_corners=False)).to(self.weight_dtype).repeat(2,1,1,1)
        rgb_resized8_latents = self.encode_RGB(F.interpolate(input_rgb, size=input_rgb.shape[-1]//8, mode='bilinear', align_corners=False)).to(self.weight_dtype).repeat(2,1,1,1)

        # batched empty text embedding
        if self.empty_text_embed is None:
            self.__encode_empty_text()
            
        batch_empty_text_embed = self.empty_text_embed.repeat((bsz, 1, 1))  # [B, 2, 1024]

        # batch discriminative embedding
        discriminative_label = torch.tensor([[0, 1], [1, 0]], dtype=self.weight_dtype, device='cuda')
        BDE = torch.cat([torch.sin(discriminative_label), torch.cos(discriminative_label)], dim=-1).repeat_interleave(bsz, 0)

        # denoising loop
        if show_pbar:
            iterable = tqdm(
                enumerate(timesteps),
                total=len(timesteps),
                leave=False,
                desc=" " * 4 + "Diffusion denoising",
            )
        else:
            iterable = enumerate(timesteps)

        for i, t in iterable:
            unet_input = torch.cat([rgb_latent, mask_edge_latent], dim=1)  # this order is important: [1,8,H,W]
            noise_pred = self.unet(unet_input, t.repeat(2), encoder_hidden_states=batch_empty_text_embed.repeat(2,1,1), class_labels = BDE,\
                               rgb_token=[rgb_latent, rgb_resized2_latents, rgb_resized4_latents, rgb_resized8_latents],\
                ).sample  # [B, 4, h, w]

            # compute x_T -> x_0
            mask_edge_latent = self.scheduler.step(noise_pred, t, mask_edge_latent).prev_sample
        

        mask, edge = self.decode(mask_edge_latent)

        mask = torch.clip(mask, -1.0, 1.0)
        mask = (mask + 1.0) / 2.0

        edge = torch.clip(edge, -1.0, 1.0)
        edge = (edge + 1.0) / 2.0
        return mask, edge
        
    
    def encode_RGB(self, rgb_in: torch.Tensor) -> torch.Tensor:

        # encode
        h = self.vae.encoder(rgb_in)
        moments = self.vae.quant_conv(h)
        mean, logvar = torch.chunk(moments, 2, dim=1)
        # scale latent
        rgb_latent = mean * self.rgb_latent_scale_factor
        
        return rgb_latent
    
    
    def decode(self, mask_edge_latent: torch.Tensor) -> torch.Tensor:

        # scale latents
        mask_edge_latent = mask_edge_latent / self.mask_latent_scale_factor
        # decode
        z = self.vae.post_quant_conv(mask_edge_latent).to(self.weight_dtype)
        stacked = self.vae.decoder(z)
        # mean of output channels
        mask_stacked, edge_stacked = torch.chunk(stacked, 2, dim=0)
        mask_mean = mask_stacked.mean(dim=1, keepdim=True)
        edge_mean = edge_stacked.mean(dim=1, keepdim=True)

        return mask_mean, edge_mean
    
