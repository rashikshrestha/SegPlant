import random
import sys
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


def train_eval(epoch, data_loader, mode='train', jitter=None):
  """
  Function to Train and Evaluate the model
  """
  model.train() if mode=='train' else model.eval()
  running_loss = 0.0
  for images, branch, leaf in tqdm(data_loader, leave=False, desc=F"{mode} e{epoch+1}"):
    images, branch, leaf = images.to(device), branch.to(device), leaf.to(device)
    masks = torch.concat((branch, leaf), dim=1)
    if jitter is not None:
      images = jitter(images)

    # for i,imm in enumerate(images):
    #   imm = to_pil_image(imm)
    #   imm.save(f"{i}.png")
    # exit()

    outputs = model(images) # Do prediction
    loss = loss_fn(outputs, masks) # Compute Loss
    running_loss += loss.item()
    if mode=='train':
      optimizer.zero_grad()
      loss.backward() # Compute gradients
      optimizer.step() # Update Model
  epoch_loss = running_loss / len(data_loader)
  print(f"Epoch [{epoch+1}/{num_epochs}],\t{mode} loss:\t{epoch_loss:.4f}")
  return epoch_loss


if __name__=='__main__':
  if len(sys.argv) < 2:
    print("Usage: python train_unet.py /path/to/dataset")
    exit()
  
  print("Training Script started.")

  # ----------- Parameters -------------------
  dataset_dir = sys.argv[1]
  # dataset_dir = "/home/rashik_shrestha/dataset/blender_plants"
  seed = 0
  device = 'cuda'
  start_epoch = 0
  num_epochs = 100
  learning_rate = 0.001
  batch_size = 4
  resize = (640,480)
  # ---------------------------------------------

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

  # Model
  model = UNet(dimensions=2).to(device)
  # model.load_state_dict(torch.load('models/e30_0.169.pt', weights_only=True))

  # Loss and Optimizer
  loss_fn = DiceLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

  # Training Starts
  best_loss = 1
  train_losses = []
  eval_losses = []
  for epoch in range(start_epoch, start_epoch+num_epochs):
    # Train and Evalutate
    train_loss = train_eval(epoch, train_loader, 'train', jitter=jitter_transform)
    eval_loss = train_eval(epoch, eval_loader, 'eval', jitter=jitter_transform)
    
    # Accumulate losses
    train_losses.append(train_loss)
    eval_losses.append(eval_loss)
    
    # Save best model
    if eval_loss < best_loss:
      best_loss = eval_loss
      print(f"Best Model is of epoch {epoch+1}")
      torch.save(model.state_dict(), f"models/e{epoch+1}_{eval_loss:.3f}.pt")
