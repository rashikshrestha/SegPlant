import os
from pathlib import Path
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor


class BlenderPlantsDataset(Dataset):
    """
    Wrapper to BlenderPlants dataset.
    It has 5637 high quality renders of a blender plant from random viewpoint and camera focal length.
    Among which first 4510 (80%) is the Train split, and last 1127 (20%) is the Evaluation split.
    """
    def __init__(self, path, split='train', resize=None, get_mask=True):
      self.path = Path(path)
      self.resize = resize
      self.get_mask = get_mask
      
      # Get idx as per the split type
      if split == 'train':
        self.idx = np.arange(0,4510)
      elif split == 'eval':
        self.idx = np.arange(4510, 5637)
      else:
        files = os.listdir(self.path/'images')
        self.idx = []
        for file in files:
          self.idx.append(int(file.split('.')[0]))
        self.idx.sort()
        self.idx = np.array(self.idx)
         
      # self.idx = np.arange(0,45,1) if split == 'train' else np.arange(4501, 4600)
        
    def __len__(self):
      return len(self.idx)
        
    def __getitem__(self, idx):
      img_id = self.idx[idx]
      # Read Image
      img_path = self.path/'images'/f"{img_id:05d}.png"
      img = Image.open(img_path)
      if self.resize is not None: img = img.resize(self.resize)
      if img.mode != 'RGB': img = img.convert('RGB')
      img = pil_to_tensor(img)/255.0
      img = img.type(torch.float32)

      if not self.get_mask:
         return img
     
      # Read branch mask
      branch_path = self.path/'masks'/'branch'/f"{img_id:05d}_1.png"
      branch = Image.open(branch_path)
      if self.resize is not None: branch = branch.resize(self.resize)
      branch = branch.convert('L')
      branch = np.array(branch)
      branch = np.where(branch>200, 1.0, 0.0)
      branch = torch.as_tensor(branch, dtype=torch.float32)[None]

      # Read leaf mask
      leaf_path = self.path/'masks'/'leaf'/f"{img_id:05d}_1.png"
      leaf = Image.open(leaf_path)
      if self.resize is not None: leaf = leaf.resize(self.resize)
      leaf = leaf.convert('L')
      leaf = np.array(leaf)
      leaf = np.where(leaf>200, 1.0, 0.0)
      leaf = torch.as_tensor(leaf, dtype=torch.float32)[None]
      
      return img, branch, leaf