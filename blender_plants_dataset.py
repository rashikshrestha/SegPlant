import os
from pathlib import Path
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor
import utils


class BlenderPlantsDataset(Dataset):
    """
    Wrapper to BlenderPlants dataset.
    It has 5637 high quality renders of a blender plant from random viewpoint and camera focal length.
    Among which first 4510 (80%) is the Train split, and last 1127 (20%) is the Evaluation split.
    """
    def __init__(self, path, split='train', resize=None, get_mask=True, get_points=False, overwrite_mask_shape=False):
      self.path = Path(path)
      self.resize = resize
      self.get_mask = get_mask
      self.get_points = get_points
      
      if overwrite_mask_shape:
        self.mask_resize = (256,256)
      else:
        self.mask_resize = self.resize
      
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
      
      if self.get_mask: 
        # Read branch mask
        branch_path = self.path/'masks'/'branch'/f"{img_id:05d}_1.png"
        branch = Image.open(branch_path)
        if self.resize is not None: branch = branch.resize(self.mask_resize)
        branch = branch.convert('L')
        branch = np.array(branch)
        branch = np.where(branch>200, 1.0, 0.0)
        branch = torch.as_tensor(branch, dtype=torch.float32)[None]

        # Read leaf mask
        leaf_path = self.path/'masks'/'leaf'/f"{img_id:05d}_1.png"
        leaf = Image.open(leaf_path)
        if self.resize is not None: leaf = leaf.resize(self.mask_resize)
        leaf = leaf.convert('L')
        leaf = np.array(leaf)
        leaf = np.where(leaf>200, 1.0, 0.0)
        leaf = torch.as_tensor(leaf, dtype=torch.float32)[None]

      if self.get_points:
        # Branch Points
        branch_points_path = self.path/'points'/'branch'/f"{img_id:05d}.txt"
        if branch_points_path.exists():
          branch_points = torch.as_tensor(np.loadtxt(branch_points_path, delimiter=','))
        else:
          branch_points = utils.mask_to_points(branch)
         
        # Leaf Points 
        leaf_points_path = self.path/'points'/'leaf'/f"{img_id:05d}.txt"
        if leaf_points_path.exists():
          leaf_points = torch.as_tensor(np.loadtxt(leaf_points_path, delimiter=','))
        else:
          leaf_points = utils.mask_to_points(leaf)
        
        points = torch.stack((branch_points, leaf_points), dim=0)
        
      if not self.get_points and not self.get_mask:
        return img
      elif not self.get_points and self.get_mask:
        return img, branch, leaf
      elif self.get_points and not self.get_mask:
        return img, points
      else:
        return img, branch, leaf, points
      
  
if __name__=='__main__':
  # dataset = BlenderPlantsDataset('test_data/blender_plants', 'test', resize=(640,480), get_mask=True, get_points=True)
  dataset = BlenderPlantsDataset('/home/rashik_shrestha/dataset/blender_plants', 'eval', resize=(640,480), get_mask=True, get_points=True)
  dp = dataset[0]
  