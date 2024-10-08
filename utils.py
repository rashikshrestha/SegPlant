import numpy as np
import torch
from torchvision.transforms.functional import to_pil_image
from PIL import Image
import random
import matplotlib.pyplot as plt

def tensor_to_numpy_image(img_tensor):
    """
    Convert pytorch tensor (0-1, float) to numpy image (0-255, uint8)
    """
    return np.array(to_pil_image(img_tensor)).astype(np.uint8)

def mask_overlay(image, branch, leaf):
    """
    Parameters
    ---------
    image: torch.Tensor
        (3, H, W) range of 0-1
    branch, leaf: torch.Tensor
        (1, H, W) range of 0-1
   
    Returns
    -------
    masked_image: np.array
        (3, H, W) range 0-255
    """
    th = 0.9
    branch_mask = torch.where(branch>th, 1.0, 0.0)
    leaf_mask = torch.where(leaf>th, 1.0, 0.0)

    padding = torch.zeros_like(branch_mask)
    
    branch_mask_rgb = torch.concat((padding, padding, branch_mask), dim=0)
    leaf_mask_rgb = torch.concat((leaf_mask, padding, padding), dim=0)
    
    # Image Tensor to Numpy
    image_np = tensor_to_numpy_image(image) 
    branch_mask_rgb_np = tensor_to_numpy_image(branch_mask_rgb)
    leaf_mask_rgb_np = tensor_to_numpy_image(leaf_mask_rgb)
    
    # Mask
    image_masked = image_np.astype(np.uint64) + branch_mask_rgb_np.astype(np.uint64) + leaf_mask_rgb_np.astype(np.uint64)
    image_masked = np.clip(image_masked, 0, 255).astype(np.uint8)
    image_masked_pil = Image.fromarray(image_masked)
    image_pil = Image.fromarray(image_np)
   
    # Blend Images 
    alpha = 0.3
    blended_image = Image.blend(image_masked_pil, image_pil, alpha)
    
    return np.array(blended_image)

def single_mask_overlay(image, branch, type='branch'):
    """
    Parameters
    ---------
    image: torch.Tensor
        (3, H, W) range of 0-1
    branch, leaf: torch.Tensor
        (1, H, W) range of 0-1
   
    Returns
    -------
    masked_image: np.array
        (3, H, W) range 0-255
    """
    th = 0.9
    branch_mask = torch.where(branch>th, 1.0, 0.0)

    padding = torch.zeros_like(branch_mask)
   
    if type=='branch':
        branch_mask_rgb = torch.concat((padding, padding, branch_mask), dim=0)
    else:
        branch_mask_rgb = torch.concat((branch_mask, padding, padding), dim=0)
        
    
    # Image Tensor to Numpy
    image_np = tensor_to_numpy_image(image) 
    branch_mask_rgb_np = tensor_to_numpy_image(branch_mask_rgb)
    
    # Mask
    image_masked = image_np.astype(np.uint64) + branch_mask_rgb_np.astype(np.uint64)
    image_masked = np.clip(image_masked, 0, 255).astype(np.uint8)
    image_masked_pil = Image.fromarray(image_masked)
    image_pil = Image.fromarray(image_np)
   
    # Blend Images 
    alpha = 0.3
    blended_image = Image.blend(image_masked_pil, image_pil, alpha)
    
    return np.array(blended_image)


def mask_to_points(mask, num=3):
    segmented_points = torch.argwhere(mask == 1)
    if len(segmented_points) > num:
        selected_indices = random.sample(range(len(segmented_points)), num)
        selected_points = segmented_points[selected_indices]
        selected_points = selected_points[:,1:]
        # print(selected_points)
        
        selected_points[:,[0,1]] = selected_points[:,[1,0]]

        # print(selected_points.shape)
        return selected_points
    else:
        return torch.zeros((3,2))

def plot_points(image, points):
    image_np = image.permute(1, 2, 0).cpu().numpy()*255.
    image_np = image_np.astype(np.uint8)
    plt.imshow(image_np)
    
    points_np = points.cpu().numpy()
    
    plt.scatter(points_np[0,:,1], points_np[0,:,0], c='blue', marker='x', label='branch')
    plt.scatter(points_np[1,:,1], points_np[1,:,0], c='red', marker='.', label='leaf')
    plt.legend()
    plt.xticks([])
    plt.yticks([])
    plt.savefig('plots.png', bbox_inches='tight', pad_inches=0)
