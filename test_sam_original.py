import random
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from PIL import Image

from transformers import SamModel, SamProcessor

# My Modules
import utils
from blender_plants_dataset import BlenderPlantsDataset
from segmentation_loss import DiceLoss


def sam_predict(images, points, idx):
  points = points[:,idx,:,:].cpu()
  
  model_inputs = processor(images=images, input_points=points, return_tensors="pt", do_rescale=False).to(device)
  outputs = model(**model_inputs)
  # print("got output")
  masks = processor.image_processor.post_process_masks(
    outputs.pred_masks.cpu(), model_inputs["original_sizes"].cpu(), model_inputs["reshaped_input_sizes"].cpu()
  )
  mask = masks[0][0].type(torch.float32)
  # print(mask.shape, mask.min(), mask.max())
  pred_mask = mask[1][None]
 
  #! Save the mask 
  # print('vis')
  # print(pred_mask.shape)
  # mask_np = (pred_mask*255).numpy().astype(np.uint8)[0]
  # Image.fromarray(mask_np).save('haha.png')
  

  return pred_mask



if __name__=='__main__':
  print("Training Script started.")

  # Parameters
  seed = 0
  idx = 1 # 0 for branch, 1 for leaf
  device = 'cuda'
  batch_size = 1
  resize = (640,480)
  dataset_dir = "/home/rashik_shrestha/dataset/blender_plants"
  model_path = "facebook/sam-vit-huge"

  # Random Seed
  torch.manual_seed(seed)
  np.random.seed(seed)
  random.seed(seed)

  # Dataset
  eval_dataset = BlenderPlantsDataset(dataset_dir, split='eval', resize=resize, get_points=True)
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
  for images, branch, leaf, points in tqdm(eval_loader, desc=F"Evaluating"):
    images, branch, leaf, points = images.to(device), branch.to(device), leaf.to(device), points.to(device)
    # print(images.shape, branch.shape, leaf.shape, points.shape)
   
    with torch.no_grad(): 
      pred_branch_mask = sam_predict(images, points, idx=idx)
    
    pred_branch_mask_cpu = pred_branch_mask.cpu()
    
    # print(pred_branch_mask_cpu.shape, branch[0].shape)

    loss, dice, iou = loss_fn(pred_branch_mask_cpu, branch[0].cpu())

    # Gather Metrics
    running_loss += loss.item()
    running_dice += dice.item()
    running_iou += iou.item()
   
  running_loss /= len(eval_loader)
  running_dice /= len(eval_loader)
  running_iou /= len(eval_loader)
  
  print("Statistics")
  print("----------")
  print(f"Loss: {running_loss}")
  print(f"Dice: {running_dice}")
  print(f"IoU: {running_iou}")
    