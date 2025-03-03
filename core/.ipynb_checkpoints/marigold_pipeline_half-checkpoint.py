
from typing import Any, Dict, Union

import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm.auto import tqdm
from PIL import Image
from diffusers import (
    DiffusionPipeline,
    DDIMScheduler,
    UNet2DConditionModel,
    AutoencoderKL,
)
from diffusers.utils import BaseOutput
from transformers import CLIPTextModel, CLIPTokenizer

from utils.image_util import resize_max_res,chw2hwc,colorize_depth_maps
from utils.colormap import kitti_colormap
from utils.depth_ensemble import ensemble_depths



class DepthPipelineOutput(BaseOutput):
    """
    Output class for Marigold monocular depth prediction pipeline.

    Args:
        depth_np (`np.ndarray`):
            Predicted depth map, with depth values in the range of [0, 1].
        depth_colored (`PIL.Image.Image`):
            Colorized depth map, with the shape of [3, H, W] and values in [0, 1].
        uncertainty (`None` or `np.ndarray`):
            Uncalibrated uncertainty(MAD, median absolute deviation) coming from ensembling.
    """
    depth_np: np.ndarray
    depth_colored: Image.Image
    uncertainty: Union[None, np.ndarray]


class DepthEstimationPipeline(DiffusionPipeline):
    # two hyper-parameters
    rgb_latent_scale_factor = 0.18215
    depth_latent_scale_factor = 0.18215
    
    def __init__(self,
                 unet:UNet2DConditionModel,
                 vae:AutoencoderKL,
                 scheduler:DDIMScheduler,
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
                 processing_res: int = 768,
                 match_input_res:bool =True,
                 batch_size:int =0,
                 color_map: str="Spectral",
                 show_progress_bar:bool = True,
                 ensemble_kwargs: Dict = None,
                 ) -> DepthPipelineOutput:
        
        # inherit from thea Diffusion Pipeline
        device = self.device
        input_size = input_image.size
        
        # adjust the input resolution.
        if not match_input_res:
            assert (
                processing_res is not None                
            )," Value Error: `resize_output_back` is only valid with "
        
        assert processing_res >=0
        assert denosing_steps >=1
        assert ensemble_size >=1
        
        # --------------- Image Processing ------------------------
        # Resize image
        if processing_res >0:
            input_image = resize_max_res(
                input_image, max_edge_resolution=processing_res
            ) # resize image: for kitti is 231, 768
        
        
        # Convert the image to RGB, to 1. reomve the alpha channel.
        input_image = input_image.convert("RGB")
        image = np.array(input_image)
        

        # Normalize RGB Values.
        rgb = np.transpose(image,(2,0,1))
        rgb_norm = rgb / 255.0 * 2 - 1.
        rgb_norm = torch.from_numpy(rgb_norm).to(self.dtype)
        rgb_norm = rgb_norm.to(device)
        
        rgb_norm = rgb_norm.half()

        assert rgb_norm.min() >= -1.0 and rgb_norm.max() <= 1.0
        
        # ----------------- predicting depth -----------------
        duplicated_rgb = torch.stack([rgb_norm] * ensemble_size)
        single_rgb_dataset = TensorDataset(duplicated_rgb)
        
        # find the batch size
        if batch_size>0:
            _bs = batch_size
        else:
            _bs = 1
        
        single_rgb_loader = DataLoader(single_rgb_dataset,batch_size=_bs,shuffle=False)
        
        # predicted the depth
        depth_pred_ls = []
        
        if show_progress_bar:
            iterable_bar = tqdm(
                single_rgb_loader, desc=" " * 2 + "Inference batches", leave=False
            )
        else:
            iterable_bar = single_rgb_loader
        
        for batch in iterable_bar:
            (batched_image,)= batch  # here the image is still around 0-1
            depth_pred_raw = self.single_infer(
                input_rgb=batched_image,
                num_inference_steps=denosing_steps,
                show_pbar=show_progress_bar,
            )
            depth_pred_ls.append(depth_pred_raw.detach().clone())
        
        depth_preds = torch.concat(depth_pred_ls, axis=0).squeeze() #(10,224,768)
        torch.cuda.empty_cache()  # clear vram cache for ensembling
        

        # ----------------- Test-time ensembling -----------------
        if ensemble_size > 1:
            depth_pred, pred_uncert = ensemble_depths(
                depth_preds, **(ensemble_kwargs or {})
            )
        else:
            depth_pred = depth_preds
            pred_uncert = None
        
        # ----------------- Post processing -----------------
        # Scale prediction to [0, 1]
        min_d = torch.min(depth_pred)
        max_d = torch.max(depth_pred)
        depth_pred = (depth_pred - min_d) / (max_d - min_d)
        
        # Convert to numpy
        depth_pred = depth_pred.cpu().numpy().astype(np.float32)

        # Resize back to original resolution
        if match_input_res:
            pred_img = Image.fromarray(depth_pred)
            pred_img = pred_img.resize(input_size)
            depth_pred = np.asarray(pred_img)

        # Clip output range: current size is the original size
        depth_pred = depth_pred.clip(0, 1)
        
        # colorization using the KITTI Color Plan.
        depth_pred_vis = depth_pred * 70
        disp_vis = 400/(depth_pred_vis+1e-3)
        disp_vis = disp_vis.clip(0,500)
        
        depth_color_pred =  kitti_colormap(disp_vis)
    
        # Colorize
        depth_colored = colorize_depth_maps(
            depth_pred, 0, 1, cmap=color_map
        ).squeeze()  # [3, H, W], value in (0, 1)
        depth_colored = (depth_colored * 255).astype(np.uint8)
        depth_colored_hwc = chw2hwc(depth_colored)
        depth_colored_img = Image.fromarray(depth_colored_hwc)

        
        return DepthPipelineOutput(
            depth_np = depth_pred,
            depth_colored = depth_colored_img,
            uncertainty=pred_uncert,
        )
        
    
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
        # print(text_input_ids.shape)
        self.empty_text_embed = self.text_encoder(text_input_ids)[0].to(self.dtype) #[1,2,1024]
        self.empty_text_embed = self.empty_text_embed.half()

        
    @torch.no_grad()
    def single_infer(self,input_rgb:torch.Tensor,
                     num_inference_steps:int,
                     show_pbar:bool,):
        
        
        device = input_rgb.device
        
        # Set timesteps: inherit from the diffuison pipeline
        self.scheduler.set_timesteps(num_inference_steps, device=device) # here the numbers of the steps is only 10.
        timesteps = self.scheduler.timesteps  # [T]
        
        # encode image
        rgb_latent = self.encode_RGB(input_rgb) # 1/8 Resolution with a channel nums of 4. 
        
        
        # Initial depth map (Guassian noise)
        depth_latent = torch.randn(
            rgb_latent.shape, device=device, dtype=self.dtype
        )  # [B, 4, H/8, W/8]
        
        depth_latent = depth_latent.half()

        # Batched empty text embedding
        if self.empty_text_embed is None:
            self.__encode_empty_text()
            
        batch_empty_text_embed = self.empty_text_embed.repeat(
            (rgb_latent.shape[0], 1, 1)
        )  # [B, 2, 1024]
        
        # Denoising loop
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
            unet_input = torch.cat(
                [rgb_latent, depth_latent], dim=1
            )  # this order is important: [1,8,H,W]
            
            # print(unet_input.shape)

            # predict the noise residual
            noise_pred = self.unet(
                unet_input, t, encoder_hidden_states=batch_empty_text_embed
            ).sample  # [B, 4, h, w]

            # compute the previous noisy sample x_t -> x_t-1
            depth_latent = self.scheduler.step(noise_pred, t, depth_latent).prev_sample
        
        torch.cuda.empty_cache()
        depth = self.decode_depth(depth_latent)
        # clip prediction
        depth = torch.clip(depth, -1.0, 1.0)
        # shift to [0, 1]
        depth = (depth + 1.0) / 2.0
        return depth
        
    
    def encode_RGB(self, rgb_in: torch.Tensor) -> torch.Tensor:
        """
        Encode RGB image into latent.

        Args:
            rgb_in (`torch.Tensor`):
                Input RGB image to be encoded.

        Returns:
            `torch.Tensor`: Image latent.
        """

        
        # encode
        h = self.vae.encoder(rgb_in)

        moments = self.vae.quant_conv(h)
        mean, logvar = torch.chunk(moments, 2, dim=1)
        # scale latent
        rgb_latent = mean * self.rgb_latent_scale_factor
        
        return rgb_latent
    
    def decode_depth(self, depth_latent: torch.Tensor) -> torch.Tensor:
        """
        Decode depth latent into depth map.

        Args:
            depth_latent (`torch.Tensor`):
                Depth latent to be decoded.

        Returns:
            `torch.Tensor`: Decoded depth map.
        """
        # scale latent
        depth_latent = depth_latent / self.depth_latent_scale_factor
        
        depth_latent = depth_latent.half()
        # decode
        z = self.vae.post_quant_conv(depth_latent)
        stacked = self.vae.decoder(z)
        # mean of output channels
        depth_mean = stacked.mean(dim=1, keepdim=True)
        return depth_mean


