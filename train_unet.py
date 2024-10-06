import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# My Modules
from unet import UNet
from blender_plants_dataset import BlenderPlantsDataset
from segmentation_loss import DiceLoss

# Parameters
seed = 0
device = 'cuda'
start_epoch = 0
num_epochs = 100
learning_rate = 0.001
batch_size = 4

# Random Seed
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

#! Dataset
train_dataset = BlenderPlantsDataset('/home/rashik_shrestha/workspace/3d_models/data', split='train')
eval_dataset = BlenderPlantsDataset('/home/rashik_shrestha/workspace/3d_models/data', split='eval')
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=True)

print(len(train_dataset))
print(len(eval_dataset))

exit()

#! Model
model = UNet(dimensions=1).to(device)
# model.load_state_dict(torch.load('models/e30_0.169.pt', weights_only=True))

#! Loss, Optimizer
loss_fn = DiceLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

def train():
  model.train()
  running_loss = 0.0
  for images, masks in tqdm(train_loader, desc="Training"):
    images, masks = images.to(device), masks.to(device)
    outputs = model(images)
    loss = loss_fn(outputs, masks)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    running_loss += loss.item()
  train_epoch_loss = running_loss / len(train_loader) 
  return train_epoch_loss

def test():
  model.eval()
  running_loss = 0.0
  for images, masks in tqdm(train_loader, desc="Testing"):
    images, masks = images.to(device), masks.to(device)
    outputs = model(images)
    loss = loss_fn(outputs, masks)
    running_loss += loss.item() 
  test_epoch_loss = running_loss / len(train_loader)
  return test_epoch_loss

#! Train Loop
for epoch in range(start_epoch, start_epoch+num_epochs):

  #! --- Train ---
  train_epoch_loss = train()
  
  #! --- Test ---
  test_epoch_loss = train_epoch_loss
  # test_epoch_loss = test()

  
  print(f"Epoch [{epoch+1}/{start_epoch+num_epochs}], Train Loss: {train_epoch_loss:.4f}, Test Loss: {test_epoch_loss:.4f}")
  
  torch.save(model.state_dict(), f"plant_models/e{epoch+1}_{test_epoch_loss:.3f}.pt")