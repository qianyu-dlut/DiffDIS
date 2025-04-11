import torch
import torch.nn.functional as F
import  os, argparse
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
from datetime import datetime
from diffusers import (
    DDPMScheduler,
    UNet2DConditionModel_diffdis,
    AutoencoderKL,
)
from transformers import CLIPTextModel, CLIPTokenizer
from utils.dataset_strategy import get_loader
from utils.utils import *
from utils.image_util import cutmix
from torch.cuda import amp
from safetensors.torch import save_model

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('./runs/DiffDIS')

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=100, help='epoch number')
parser.add_argument('--lr_gen', type=float, default=3e-5, help='learning rate')
parser.add_argument('--batchsize', type=int, default=4, help='training batch size')
parser.add_argument('--trainsize', type=int, default=1024, help='training dataset size')
parser.add_argument('--decay_rate', type=float, default=0.95, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=30, help='every n epochs decay learning rate')
parser.add_argument("--pretrained_model_name_or_path", type=str, default='/path/to/sd-turbo/')
parser.add_argument("--dataset_path", type=str, default='/path/to/DIS5K/')


opt = parser.parse_args()
print('Generator Learning Rate: {}'.format(opt.lr_gen))
# build models
text_encoder = CLIPTextModel.from_pretrained(opt.pretrained_model_name_or_path, subfolder='text_encoder')
vae = AutoencoderKL.from_pretrained(opt.pretrained_model_name_or_path, subfolder='vae')
unet = UNet2DConditionModel_diffdis.from_pretrained(opt.pretrained_model_name_or_path, subfolder="unet",
                                    in_channels=4, sample_size=96,
                                    low_cpu_mem_usage=False,
                                    ignore_mismatched_sizes=False,
                                    class_embed_type='projection',
                                    projection_class_embeddings_input_dim=4,
                                    mid_extra_cross=True,
                                    mode = 'DBIA',
                                    use_swci = True, 
                                    )


text_encoder.requires_grad_(False)
vae.requires_grad_(False)
unet = replace_unet_conv_in(unet)
unet = update_att_weights(unet) 
unet.train().cuda()


noise_scheduler = DDPMScheduler.from_pretrained(opt.pretrained_model_name_or_path, subfolder='scheduler')##{'clip_sample_range', 'rescale_betas_zero_snr', 'sample_max_value', 'timestep_spacing', 'thresholding', 'dynamic_thresholding_ratio'} 
noise_scheduler.set_timesteps(1, device="cuda")
noise_scheduler.alphas_cumprod = noise_scheduler.alphas_cumprod.cuda()
tokenizer = CLIPTokenizer.from_pretrained(opt.pretrained_model_name_or_path,subfolder='tokenizer')


params, params_class_embedding = [], []
for name, param in unet.named_parameters():
    if 'class_embedding' in name:
        params_class_embedding.append(param)
    else:
        params.append(param)
        
generator_optimizer = torch.optim.Adam([
            {"params": params, "lr":opt.lr_gen},
            {"params": params_class_embedding, "lr":opt.lr_gen*10}
        ])

# load data
image_root = f'{opt.dataset_path}/DIS-TR/im/'
gt_root = f'{opt.dataset_path}/DIS-TR/gt/'
edge_root = f'{opt.dataset_path}/DIS-TR/contour/'

train_loader = get_loader(image_root, gt_root, edge_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
total_step = len(train_loader)
print(total_step)

rgb_latent_scale_factor = 0.18215
weight_dtype = torch.float32
text_encoder.to('cuda', dtype=weight_dtype)
vae.to('cuda', dtype=weight_dtype)

mse_loss = torch.nn.MSELoss(size_average=True, reduce=True)
size_rates = [1] 
scaler = amp.GradScaler(enabled=True)


for epoch in range(1, opt.epoch+1):
    unet.train()
    loss_record = AvgMeter()
    print('Generator Learning Rate: {}'.format(generator_optimizer.param_groups[0]['lr']))
    for i, pack in enumerate(train_loader, start=1):
        for rate in size_rates:
            generator_optimizer.zero_grad()

            rgb, label, edge, box = pack

            rgb = rgb.cuda().to(weight_dtype) 
            label=label.unsqueeze(1).repeat(1,3,1,1).cuda().to(weight_dtype) 
            edge=edge.unsqueeze(1).repeat(1,3,1,1).cuda().to(weight_dtype) 
            box=box.unsqueeze(1).repeat(1,3,1,1).cuda().to(weight_dtype) 

            bsz = rgb.shape[0]
            assert bsz % 2 == 0, "Batch size must be even"

            rgb_chunks = rgb.chunk(bsz // 2, dim=0)       
            label_chunks = label.chunk(bsz // 2, dim=0)    
            edge_chunks = edge.chunk(bsz // 2, dim=0)      
            box_chunks = box.chunk(bsz // 2, dim=0) 

            # apply cutmix within each chunk
            rgbs, labels, edges = [], [], []
            for rgb_, label_, edge_, box_ in zip(rgb_chunks, label_chunks, edge_chunks, box_chunks):
                mixed_rgb, mixed_label, mixed_edge = cutmix(rgb_, label_, edge_, box_)
                rgbs.append(mixed_rgb)
                labels.append(mixed_label)
                edges.append(mixed_edge)
            
            rgb_mix = torch.cat(rgbs, dim=0)
            label_mix = torch.cat(labels, dim=0)
            edge_mix = torch.cat(edges, dim=0)

            # map pixels into latent space
            h_batch = vae.encoder(torch.cat((rgb_mix, label_mix, edge_mix), dim=0).to(weight_dtype))
            moments_batch = vae.quant_conv(h_batch)
            mean_batch, logvar_batch = torch.chunk(moments_batch, 2, dim=1)
            batch_latents = mean_batch * rgb_latent_scale_factor
            rgb_latents, mask_latents, edge_latents = torch.chunk(batch_latents, 3, dim=0)
            
            # generate multi-scale conditions
            rgb_resized2_latents, rgb_resized4_latents, rgb_resized8_latents = generate_multi_scale_latents(rgb_mix, rgb_latent_scale_factor, vae, weight_dtype, opt)
            
            # concat mask and edge latents along batch dimension 
            unified_latents = torch.cat((mask_latents,edge_latents), dim=0)

            # create multi-resolution noise
            noise = pyramid_noise_like(unified_latents, discount=0.8) 

            # set timestep to T
            timesteps = torch.tensor([999], device="cuda").long()
            
            # add noise 
            noisy_unified_latents = noise_scheduler.add_noise(unified_latents, noise, timesteps.repeat(bsz*2))
 
            # encode text embedding for empty prompt
            prompt = ""
            text_inputs =tokenizer(
                prompt,
                padding="do_not_pad",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids.to(text_encoder.device) 
            empty_text_embed = text_encoder(text_input_ids)[0].to(weight_dtype)
            batch_empty_text_embed = empty_text_embed.repeat((noisy_unified_latents.shape[0], 1, 1))  

            # batch discriminative embedding
            discriminative_label = torch.tensor([[0, 1], [1, 0]], dtype=weight_dtype, device='cuda')
            BDE = torch.cat([torch.sin(discriminative_label), torch.cos(discriminative_label)], dim=-1).repeat_interleave(bsz, 0)
            unet_input = torch.cat([rgb_latents.repeat(2,1,1,1),noisy_unified_latents], dim=1)  
            
            # predict the noise 
            noise_pred = unet(unet_input, timesteps.repeat(bsz*2), encoder_hidden_states=batch_empty_text_embed, class_labels = BDE,\
                               rgb_token=[rgb_latents.repeat(2,1,1,1) , rgb_resized2_latents, rgb_resized4_latents, rgb_resized8_latents],\
                ).sample 
            
            # one-step denoising process
            x_denoised = noise_scheduler.step(noise_pred, timesteps, noisy_unified_latents, return_dict=True).prev_sample
            
            # split mask and edge latents along batch dimension 
            mask_latent, edge_latent = torch.chunk(x_denoised, 2, dim=0)

            loss1 = F.mse_loss(mask_latent.cuda().to(weight_dtype),mask_latents.cuda().to(weight_dtype), reduction="mean")
            loss2 = F.mse_loss(edge_latent.cuda().to(weight_dtype),edge_latents.cuda().to(weight_dtype), reduction="mean")
            loss = loss1 + loss2
            writer.add_scalar('mask_loss', loss1.item(), epoch * len(train_loader) + i)
            writer.add_scalar('edge_loss', loss2.item(), epoch * len(train_loader) + i)
            writer.add_scalar('total_loss', loss.item(), epoch * len(train_loader) + i)

            generator_optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(generator_optimizer)
            scaler.update()
            if rate == 1:
                loss_record.update(loss.data, opt.batchsize)

        if i % 10 == 0 or i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], gen Loss: {:.4f}, mask loss:{:.4f}, edge loss:{:.4f}'.
                  format(datetime.now(), epoch, opt.epoch, i, total_step, loss_record.show(), loss1, loss2))

    adjust_lr(generator_optimizer, opt.lr_gen, epoch, opt.decay_rate, opt.decay_epoch)

    # save checkpoints every 10 epochs
    if epoch % 10 == 0: 
        save_path = f'../saved_model/DiffDIS/Model_{epoch}/unet/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_model(unet, f'{save_path}diffusion_pytorch_model.safetensors')
        optimizer_state = generator_optimizer.state_dict()
        torch.save(optimizer_state, f'../saved_model/DiffDIS/Model_{epoch}/generator_optimizer.pth')