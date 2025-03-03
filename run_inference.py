import argparse
import os
import logging
import glob

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

from core.diffdis_pipeline import DiffDISPipeline
from diffusers import (
    DDPMScheduler,
    UNet2DConditionModel_diffdis,
    AutoencoderKL,
)
from transformers import CLIPTextModel, CLIPTokenizer

from utils.seed_all import seed_all 
from utils.utils import check_mkdir
from utils.config import diste1,diste2,diste3,diste4,disvd
from utils.image_util import resize_res

from torchvision import transforms
to_pil = transforms.ToPILImage()


## DIS dataset
to_test ={
    'DIS-VD':disvd,
    'DIS-TE1':diste1,
    'DIS-TE2':diste2,
    'DIS-TE3':diste3,
    'DIS-TE4':diste4,
}


EXTENSION_LIST = [".jpg", ".jpeg", ".png"]

if __name__=="__main__":
    
    use_seperate = False

    logging.basicConfig(level=logging.INFO)
    
    '''Set the Args'''
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, default='/path/to/your/unet/')    

    parser.add_argument("--pretrained_model_path", type=str, default='/path/to/pretrained/models/')  
      
    parser.add_argument("--output_dir", type=str, default='/path/to/save/outputs/')    

    # inference setting
    parser.add_argument(
        "--denoise_steps",
        type=int,
        default=1,
        help="Diffusion denoising steps, more steps results in higher accuracy but slower inference speed.",
    )
    parser.add_argument(
        "--ensemble_size",
        type=int,
        default=1,
        help="Number of predictions to be ensembled, more inference gives better results but runs slower.",
    )
    parser.add_argument(
        "--half_precision",
        action="store_true",
        help="Run with half-precision (16-bit float), might lead to suboptimal result.",
    )

    # resolution setting
    parser.add_argument(
        "--processing_res",
        type=int,
        default=1024,
        help="Maximum resolution of processing. 0 for using input image resolution. Default: 768.",
    )
    parser.add_argument(
        "--output_processing_res",
        action="store_true",
        help="When input is resized, out put depth at resized operating resolution. Default: False.",
    )

    # other settings
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=0,
        help="Inference batch size. Default: 0 (will be set automatically).",
    )
    
    args = parser.parse_args()
    
    pretrained_model_path = args.pretrained_model_path
    checkpoint_path = args.checkpoint_path
    output_dir = args.output_dir
    denoise_steps = args.denoise_steps
    ensemble_size = args.ensemble_size
    
    if ensemble_size>15:
        logging.warning("long ensemble steps, low speed..")
    
    half_precision = args.half_precision

    processing_res = args.processing_res
    match_input_res = not args.output_processing_res

    seed = args.seed
    batch_size = args.batch_size
    check_mkdir(output_dir)
    
    if batch_size==0:
        batch_size = 1  # set default batchsize
    # -------------------- Preparation --------------------
    # Random seed
    if seed is None:
        import time
        seed = int(time.time())
    seed_all(seed)

    # -------------------- Device --------------------
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        logging.warning("CUDA is not available. Running on CPU will be slow.")
    logging.info(f"device = {device}")

    import ttach as tta
    transforms = tta.Compose(
    [
        tta.HorizontalFlip(),
        tta.Scale(scales=[0.75, 1, 1.25], interpolation='bilinear', align_corners=False),
    ])
    

    # -------------------- Model --------------------
    if half_precision:
        dtype = torch.float16
        logging.info(f"Running with half precision ({dtype}).")
    else:
        dtype = torch.float32

    vae = AutoencoderKL.from_pretrained(pretrained_model_path,subfolder='vae')
    scheduler = DDPMScheduler.from_pretrained(pretrained_model_path,subfolder='scheduler')
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path,subfolder='text_encoder')
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_path,subfolder='tokenizer')
    unet = UNet2DConditionModel_diffdis.from_pretrained(checkpoint_path,subfolder="unet",
                                    in_channels=8, sample_size=96,
                                    low_cpu_mem_usage=False,
                                    ignore_mismatched_sizes=False,
                                    class_embed_type='projection',
                                    projection_class_embeddings_input_dim=4,
                                    mid_extra_cross=True,
                                    mode = 'DBIA',
                                    use_swci = True)
    pipe = DiffDISPipeline(unet=unet,
                            vae=vae,
                            scheduler=scheduler,
                            text_encoder=text_encoder,
                            tokenizer=tokenizer)
    print("Using Seperated Modules")
    
    logging.info("loading pipeline whole successfully.")

    pipe = pipe.to(device)

    # -------------------- Inference and saving --------------------
    for name, root in to_test.items():
        rgb_root = os.path.join(root, 'im')
        rgb_filename_list = glob.glob(os.path.join(rgb_root, "*"))
        rgb_filename_list = [
            f for f in rgb_filename_list if os.path.splitext(f)[1].lower() in EXTENSION_LIST
        ]
        rgb_filename_list = sorted(rgb_filename_list)
        with torch.no_grad():
            output_dir1 = os.path.join(output_dir, name)
            check_mkdir(output_dir1)
            logging.info(f"output dir = {output_dir1}")
            for input_image_path in tqdm(rgb_filename_list, desc=f"Estimating mask", leave=True):

                input_image_pil = Image.open(input_image_path)
                pred_name_base = os.path.splitext(os.path.basename(input_image_path))[0]

                w_,h_ = input_image_pil.size
                img_resize = resize_res(input_image_pil, resolution=processing_res)

                input_image = img_resize.convert("RGB")
                image = np.array(input_image)
                rgb = np.transpose(image,(2,0,1))
                rgb_norm = rgb / 255.0 * 2.0 - 1.0
                rgb_norm = torch.from_numpy(rgb_norm).to(device)
                rgb_norm = rgb_norm.to(device).float().unsqueeze(0)
                mask_m = []
                mask_e = []
                for transformer in transforms:
                    img_resize = transformer.augment_image(rgb_norm)
                    pipe_out_m, pipe_out_e = pipe(img_resize,
                        denosing_steps=denoise_steps,
                        ensemble_size= ensemble_size,
                        processing_res = processing_res,
                        match_input_res = match_input_res,
                        batch_size = batch_size,
                        show_progress_bar = True
                        )
                    deaug_mask_m = transformer.deaugment_mask(pipe_out_m.unsqueeze(0).unsqueeze(0))
                    mask_m.append(deaug_mask_m)
                    deaug_mask_e = transformer.deaugment_mask(pipe_out_e.unsqueeze(0).unsqueeze(0))
                    mask_e.append(deaug_mask_e)


                prediction_m = torch.mean(torch.stack(mask_m, dim=0), dim=0)
                prediction_e = torch.mean(torch.stack(mask_e, dim=0), dim=0)

                prediction_m = to_pil(prediction_m.data.squeeze(0).cpu())
                prediction_m = prediction_m.resize((w_, h_), Image.BILINEAR)
                
                prediction_e = to_pil(prediction_e.data.squeeze(0).cpu())
                prediction_e = prediction_e.resize((w_, h_), Image.BILINEAR)


                check_mkdir(os.path.join(output_dir1, 'mask'))
                mask_save_path = os.path.join(output_dir1, 'mask', f"{pred_name_base}.png")
                if os.path.exists(mask_save_path):
                    logging.warning(f"Existing file: '{mask_save_path}' will be overwritten")
                prediction_m.save(mask_save_path, mode="I;16")

                check_mkdir(os.path.join(output_dir1, 'edge'))
                edge_save_path = os.path.join(output_dir1,'edge', f"{pred_name_base}.png")
                if os.path.exists(edge_save_path):
                    logging.warning(f"Existing file: '{edge_save_path}' will be overwritten")
                prediction_e.save(edge_save_path, mode="I;16")