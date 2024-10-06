from pathlib import Path
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor


class BlenderPlantsDataset(Dataset):
    """
    Wrapper to BlenderPlants dataset.
    It has 5644 high quality renders of a blender plant from random viewpoint and camera focal length.
    Among which first 4516 (80%) is the Train split, and last 1128 (20%) is the Evaluation split.
    """
    def __init__(self, path, split='train'):
      self.path = Path(path)
      self.idx = np.arange(0,4515,1) if split == 'train' else np.arange(4516, 5644)
      # self.idx = np.arange(0,45,1) if split == 'train' else np.arange(4501, 4600)
        
    def __len__(self):
      return len(self.idx)
        
    def __getitem__(self, idx):
      # Read Image
      img_path = self.path/'images'/f"{idx:05d}.png"
      img = Image.open(img_path)
      if img.mode != 'RGB': img = img.convert('RGB')
      img = pil_to_tensor(img)/255.0
      img = img.type(torch.float32)
     
      # Read branch mask
      branch_path = self.path/'masks'/'branch'/f"{idx:05d}_1.png"
      branch = Image.open(branch_path)
      branch = branch.convert('L')
      branch = np.array(branch)
      branch = np.where(branch>200, 1.0, 0.0)
      branch = torch.as_tensor(branch, dtype=torch.float32)[None]

      # Read leaf mask
      leaf_path = self.path/'masks'/'leaf'/f"{idx:05d}_1.png"
      leaf = Image.open(leaf_path)
      leaf = branch.convert('L')
      leaf = np.array(leaf)
      leaf = np.where(leaf>200, 1.0, 0.0)
      leaf = torch.as_tensor(leaf, dtype=torch.float32)[None]
      
      return img, branch, leaf