import random
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image

# My Modules
from unet import UNet
from blender_plants_dataset import BlenderPlantsDataset
from segmentation_loss import DiceLoss


def evaluate(data_loader):
  """
  Function to Evaluate the model
  """
  running_loss, running_iou, running_dice = 0.0, 0.0, 0.0
  for images, branch, leaf in tqdm(data_loader, leave=False, desc=F"Evaluating"):
    images, branch, leaf = images.to(device), branch.to(device), leaf.to(device)
    masks = torch.concat((branch, leaf), dim=1)

    # for i,imm in enumerate(images):
    #   imm = to_pil_image(imm)
    #   imm.save(f"{i}.png")
    # exit()

    outputs = model(images) # Do prediction
    loss, dice, iou = loss_fn(outputs, masks) # Compute Loss
    
    # Accumulate Losses
    running_loss += loss.item()
    running_dice += dice.item()
    running_iou += iou.item()
    
  running_loss /= len(data_loader)
  running_dice /= len(data_loader)
  running_iou /= len(data_loader)
  
  return running_loss, running_dice, running_iou


if __name__=='__main__':
  print("Evaluation Script started.")

  # Parameters
  seed = 0
  device = 'cuda'
  batch_size = 4
  resize = (640,480)
  dataset_dir = "/home/rashik_shrestha/dataset/blender_plants"

  # Random Seed
  torch.manual_seed(seed)
  np.random.seed(seed)
  random.seed(seed)

  # Dataset
  eval_dataset = BlenderPlantsDataset(dataset_dir, split='eval', resize=resize)
  eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=True)
  print(f"Available Eval Set: {len(eval_dataset)}")

  # Model
  model = UNet(dimensions=2).to(device)
  model.load_state_dict(torch.load('models_archive/jitter/e15_0.463.pt', weights_only=True))
  model.eval()

  # Loss and Optimizer
  loss_fn = DiceLoss(get_dice_iou=True)

  # Evaluation Starts
  with torch.no_grad():
    loss, dice, iou = evaluate(eval_loader)
    
  print("Statistics")
  print("----------")
  print(f"Loss: {loss}")
  print(f"Dice: {dice}")
  print(f"IoU: {iou}")