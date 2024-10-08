from pathlib import Path
from PIL import Image
from tqdm import tqdm
import torch

# My Imports
import utils
from unet import UNet
from blender_plants_dataset import BlenderPlantsDataset
from segmentation_loss import DiceLoss


if __name__=='__main__':
    print("Testing Unet")
   
    # -------------- Parameters ---------------- 
    weights = 'models/unet.pt'
    dataset_dir = "test_data/real_plants"
    resize = (640,480)
    device = 'cuda'
    out_dir = 'output/unet'
    get_mask = False
    get_point = False
    # ------------------------------------------

    # Setup output path    
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
   
    # Test Dataset 
    test_dataset = BlenderPlantsDataset(dataset_dir, split='test', resize=resize, get_mask=get_mask, get_points=get_point)
    print(f"Test dataset len: {len(test_dataset)}")
    
    # UNet Model
    model = UNet(dimensions=2).to(device)
    model.load_state_dict(torch.load(weights, weights_only=True))
    print("Using weights: ", weights)

    # Loss fn
    loss_fn = DiceLoss(get_dice_iou=True)
    
    # Run testing 
    for i, data in tqdm(enumerate(test_dataset)):
        if get_mask:
            img, branch, leaf = data
        else:
            img = data
            
        # Save original Image
        Image.fromarray(utils.tensor_to_numpy_image(img)).save(out_dir/f"img_{i}.png")
        
        if get_mask:
            masked_img_gt = utils.mask_overlay(img, branch, leaf)
            Image.fromarray(masked_img_gt).save(out_dir/f"gt_{i}.png")
        
        # Predict Mask using Model
        img = img[None].to(device)
        out = model(img)
        pred_branch = out[0][0][None]
        pred_leaf = out[0][1][None]
        masked_img = utils.mask_overlay(img[0], pred_branch, pred_leaf)
       
        # Save pred mask 
        Image.fromarray(masked_img).save(out_dir/f"pred_{i}.png")