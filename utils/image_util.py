from PIL import Image
import torch.nn.functional as F



def resize_max_res(img: Image.Image, max_edge_resolution: int) -> Image.Image:
    """
    Resize image to limit maximum edge length while keeping aspect ratio.
    Args:
        img (`Image.Image`):
            Image to be resized.
        max_edge_resolution (`int`):
            Maximum edge length (pixel).
    Returns:
        `Image.Image`: Resized image.
    """
    
    original_width, original_height = img.size
    
    downscale_factor = min(
        max_edge_resolution / original_width, max_edge_resolution / original_height
    )
    new_width = ((int(original_width * downscale_factor) - 1) // 8 + 1) * 8
    new_height = ((int(original_height * downscale_factor) - 1) // 8 + 1) * 8
    resized_img = img.resize((new_width, new_height))
    return resized_img


def resize_max_res_tensor(input_tensor, is_disp=False, recom_resolution=768):
    original_H, original_W = input_tensor.shape[2:]
    
    downscale_factor = min(recom_resolution/original_H,
                        recom_resolution/original_W)
    
    resized_input_tensor = F.interpolate(input_tensor,
                                        scale_factor=downscale_factor,mode='bilinear',
                                        align_corners=False)
    if is_disp:
        return resized_input_tensor * downscale_factor
    else:
        return resized_input_tensor


def resize_res(img: Image.Image, resolution: int) -> Image.Image:
    
    resized_img = img.resize((resolution, resolution), Image.BILINEAR)
    return resized_img


def cutmix(data, targets, edge, boxes):
    ori_data, ori_targets, ori_edge = data[0].clone(), targets[0].clone(), edge[0].clone()
    shuf_data, shuf_targets, shuf_edge = data[1].clone(), targets[1].clone(), edge[1].clone()

    for i in [0, 1]:
        src_data = shuf_data if i == 0 else ori_data
        src_targets = shuf_targets if i == 0 else ori_targets
        src_edge = shuf_edge if i == 0 else ori_edge

        mask = boxes[i].expand_as(data[i])
        
        data[i][mask==1] = src_data[mask==1]
        targets[i][mask==1] = src_targets[mask==1]
        edge[i][mask==1] = src_edge[mask==1]

    return data, targets, edge
