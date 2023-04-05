# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 01:13:22 2023

@author: Xavier
"""

import torch
from torch.utils.data import Dataset

import pandas as pd

class DsDataset(Dataset):
 
  def __init__(self, x, y):

    self.x_train=torch.tensor(x,dtype=torch.float32)
    self.y_train=torch.tensor(y,dtype=torch.float32)
 
  def __len__(self):
    return len(self.y_train)
   
  def __getitem__(self,idx):
    return self.x_train[idx],self.y_train[idx]