import random
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image

from transformers import SamModel, SamProcessor

# My Modules
from unet import UNet
from blender_plants_dataset import BlenderPlantsDataset
from segmentation_loss import DiceLoss


def train_eval(epoch, data_loader, mode='train', jitter=None):
  """
  Function to Train and Evaluate the model
  """
  model.train() if mode=='train' else model.eval()
  running_loss = 0.0
  for images, branch, leaf in tqdm(data_loader, leave=False, desc=F"{mode} e{epoch+1}"):
    images, branch, leaf = images.to(device), branch.to(device), leaf.to(device)
    masks = torch.concat((branch, leaf), dim=1)
    masks = branch
    if jitter is not None:
      images = jitter(images)

    # for i,imm in enumerate(images):
    #   imm = to_pil_image(imm)
    #   imm.save(f"{i}.png")
    # exit()
    
    # --- Apply SAM model ---
    pil_img = [to_pil_image(img) for img in images]
    # print(pil_img[0].size)
    model_inputs = processor(images=images, return_tensors="pt", do_rescale=False).to(device)
    outputs = model(**model_inputs)
    
    pred_mask = outputs['pred_masks'].squeeze(1)
    
    pred_mask = (pred_mask-torch.min(pred_mask))/(torch.max(pred_mask)-torch.min(pred_mask))
    
    pred_mask = pred_mask[:,1,:,:]
    pred_mask = pred_mask[:,None,:,:]
    # ------------------------
    # print(pred_mask.shape, masks.shape)
    # exit()
     
    loss = loss_fn(pred_mask, masks) # Compute Loss
    running_loss += loss.item()
    if mode=='train':
      optimizer.zero_grad()
      loss.backward() # Compute gradients
      optimizer.step() # Update Model
  epoch_loss = running_loss / len(data_loader)
  print(f"Epoch [{epoch+1}/{num_epochs}],\t{mode} loss:\t{epoch_loss:.4f}")
  return epoch_loss


if __name__=='__main__':
  print("Training Script started.")

  # Parameters
  seed = 0
  device = 'cuda'
  start_epoch = 1
  num_epochs = 10
  learning_rate = 1e-5
  batch_size = 4
  resize = (640,480)
  dataset_dir = "/home/rashik_shrestha/dataset/blender_plants"
  # model_path = "facebook/sam-vit-huge"
  model_path = "models/e1_0.513"

  # Random Seed
  torch.manual_seed(seed)
  np.random.seed(seed)
  random.seed(seed)

  # Dataset
  jitter_transform = transforms.Compose([
    transforms.ColorJitter(0.0, 0.3, 0.5, 0.1)
  ])
  jitter_transform = None
  train_dataset = BlenderPlantsDataset(dataset_dir, split='train', resize=resize)
  eval_dataset = BlenderPlantsDataset(dataset_dir, split='eval', resize=resize)


  train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
  eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=True)

  print(f"Available Train Set: {len(train_dataset)}")
  print(f"Available Eval Set: {len(eval_dataset)}")

  # ---- SAM Model ----
  model = SamModel.from_pretrained(model_path).to(device)
  processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")
  print("SAM model loaded")
  
  # make sure we only compute gradients for mask decoder
  for name, param in model.named_parameters():
    if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
      param.requires_grad_(False)
  # model.load_state_dict(torch.load('models/e30_0.169.pt', weights_only=True))

  # ---- Loss and Optimizer ----
  optimizer = torch.optim.Adam(model.mask_decoder.parameters(), lr=learning_rate, weight_decay=0)
  loss_fn = DiceLoss()

  # ---- Training Starts ----
  best_loss = 1
  train_losses = []
  eval_losses = []
  for epoch in range(start_epoch, start_epoch+num_epochs):
    # Train and Evalutate
    train_loss = train_eval(epoch, train_loader, 'train', jitter=jitter_transform)
    # eval_loss = train_eval(epoch, eval_loader, 'eval', jitter=jitter_transform)
    eval_loss = train_loss
    
    # Accumulate losses
    train_losses.append(train_loss)
    eval_losses.append(eval_loss)
    
    # Save best model
    if eval_loss < best_loss:
      best_loss = eval_loss
      print(f"Best Model is of epoch {epoch+1}")
    model.save_pretrained(f"models/e{epoch+1}_{eval_loss:.3f}")
