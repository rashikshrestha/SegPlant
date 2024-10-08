import random
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from PIL import Image
from pathlib import Path
import sys

from transformers import SamModel, SamProcessor

# My Modules
import utils
from blender_plants_dataset import BlenderPlantsDataset
from segmentation_loss import DiceLoss


def sam_predict(idx, images, points=None):
  if points is not None:
    points = points[:,idx,:,:].cpu()
    model_inputs = processor(images=images, input_points=points, return_tensors="pt", do_rescale=False).to(device)
  else:
    model_inputs = processor(images=images, return_tensors="pt", do_rescale=False).to(device)
  
  outputs = model(**model_inputs)
  # print("got output")
  masks = processor.image_processor.post_process_masks(
    outputs.pred_masks.cpu(), model_inputs["original_sizes"].cpu(), model_inputs["reshaped_input_sizes"].cpu()
  )
  mask = masks[0][0].type(torch.float32)
  pred_mask = mask[1][None]
 
  # Save the mask 
  # print('vis')
  # print(pred_mask.shape)
  # mask_np = (pred_mask*255).numpy().astype(np.uint8)[0]
  # Image.fromarray(mask_np).save('haha2.png')
  # exit()
  
  return pred_mask



if __name__=='__main__':
  if len(sys.argv) < 2:
    print("Usage: python test_sam.py <0=branch, 1=leaf>")
    exit()
  print("Testing Script started.")

  # ------------- Parameters ----------------------
  idx = int(sys.argv[1]) # 0 for branch, 1 for leaf
  
  if idx==0:
    model_path = "models/sam_branch"
    vis_path = "output/sam_branch"
  else:
    model_path = "models/sam_leaf"
    vis_path = "output/sam_leaf"
    
  seed = 0
  device = 'cuda'
  batch_size = 1 # always keep 1 for test
  resize = (640,480)
  dataset_dir = "test_data/blender_plants"
  get_mask = False
  get_points = False
  vis = True
  # ------------------------------------------------------- 
  
  seg_type = 'leaf' if idx else 'branch'
  
  if vis:
    # Setup vis path
    out_dir = Path(vis_path)
    out_dir.mkdir(parents=True, exist_ok=True)

  # Random Seed
  torch.manual_seed(seed)
  np.random.seed(seed)
  random.seed(seed)

  # Dataset
  eval_dataset = BlenderPlantsDataset(dataset_dir, split='test', resize=resize, get_mask=get_mask, get_points=get_points)
  eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=True)
  print(f"Available Eval Set: {len(eval_dataset)}")

  # ---- SAM Model ----
  model = SamModel.from_pretrained(model_path).to(device)
  model.eval()
  processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")
  print("SAM model loaded")
  
  # ---- Loss and Optimizer ----
  loss_fn = DiceLoss(get_dice_iou=True)

  # ---- Evaluation Starts ----
  print('Running Evaluation') 
  running_loss, running_dice, running_iou = 0.0, 0.0, 0.0
  for i, data in enumerate(tqdm(eval_loader, desc=F"Evaluating")):
    if get_mask and get_points:
      images, branch, leaf, points = data
      images, branch, leaf, points = images.to(device), branch.to(device), leaf.to(device), points.to(device)
    elif not get_mask and get_points:
      images, points = data
      images, points = images.to(device), points.to(device)
    else:
      images = data
      images = images.to(device)
      points = None
      
    # Save original Image
    if vis: Image.fromarray(utils.tensor_to_numpy_image(images[0])).save(out_dir/f"img_{i}.png")
   
    with torch.no_grad(): 
      pred_branch_mask = sam_predict(idx, images, points)
    
    pred_branch_mask_cpu = pred_branch_mask.cpu()
    
    if vis:
      # print(pred_branch_mask_cpu.shape, images[0].shape)
      masked_img = utils.single_mask_overlay(images[0].cpu(), pred_branch_mask_cpu, type=seg_type)
      # Save pred mask 
      Image.fromarray(masked_img).save(out_dir/f"pred_{i}.png")
      # exit()
    
    if get_mask:
      if seg_type=='leaf':
        loss, dice, iou = loss_fn(pred_branch_mask_cpu, leaf[0].cpu())
      elif seg_type=='branch':
        loss, dice, iou = loss_fn(pred_branch_mask_cpu, branch[0].cpu())
        
      
      # print(loss, dice, iou)

      # Gather Metrics
      if torch.isnan(loss) or torch.isnan(dice) or torch.isnan(iou):
        print("Dealing with NaN values")
      else:
        running_loss += loss.item()
        running_dice += dice.item()
        running_iou += iou.item()
  
  if get_mask: 
    running_loss /= len(eval_loader)
    running_dice /= len(eval_loader)
    running_iou /= len(eval_loader)
    
    print("Statistics")
    print("----------")
    print(f"Loss: {running_loss}")
    print(f"Dice: {running_dice}")
    print(f"IoU: {running_iou}")
    