import torch
import torch.utils.data as data
from mat4py import loadmat
import numpy as np
import math
import random

def FindIndices(spike, state):
    data_idx = []
    for idx in range(len(spike)):
        if not math.isnan(state[idx]): # state[idx] is 0 or 1
            data_idx.append(idx)
    return data_idx

class ValidFolder(data.Dataset):
    def __init__(self, filename, history):
        self.history = history
        self.mat = loadmat(filename)
        self.spike = np.array(self.mat['trainSpike']).T.tolist()
        self.state = self.mat['trainState']
        self.data_idx = FindIndices(self.spike, self.state)
        random.shuffle(self.data_idx)
    
    def __getitem__(self, index):
        index = index % len(self.data_idx)
        data_idx = self.data_idx[index]
        cur_spike = torch.FloatTensor(self.spike[data_idx])
        cur_state = torch.LongTensor([self.state[data_idx]])

        for i in range(1, self.history):
            history_idx = self.data_idx[index] - i
            if history_idx < 0:
                history_idx = 0
            history_spike = torch.FloatTensor(self.spike[history_idx])
            cur_spike = torch.cat([cur_spike, history_spike], dim=0)
            
        return cur_spike, cur_state, data_idx
        
    def __len__(self):
        return len(self.data_idx)

def FindNanIndices(spike, state):
    data_idx = []
    for idx in range(len(spike)):
        if math.isnan(state[idx]): # state[idx] is nan
            data_idx.append(idx)
    random.shuffle(data_idx)
    return data_idx[:1500] # undersample majority class

class NanFolder(data.Dataset):
    def __init__(self, filename, history):
        self.history = history
        self.mat = loadmat(filename)
        self.spike = np.array(self.mat['trainSpike']).T.tolist()
        self.state = self.mat['trainState']
        self.valid_data_idx = FindIndices(self.spike, self.state)
        self.nan_data_idx = FindNanIndices(self.spike, self.state)
        self.data_idx = self.valid_data_idx + self.nan_data_idx
        random.shuffle(self.data_idx)
    
    def __getitem__(self, index):
        index = index % len(self.data_idx)
        data_idx = self.data_idx[index]
        cur_spike = torch.FloatTensor(self.spike[data_idx])
        if math.isnan(self.state[data_idx]):
            cur_state = torch.LongTensor([0]) # 0 - nan
        else:
            cur_state = torch.LongTensor([1]) # 1 - valid

        for i in range(1, self.history):
            history_idx = self.data_idx[index] - i
            if history_idx < 0:
                history_idx = 0
            history_spike = torch.FloatTensor(self.spike[history_idx])
            cur_spike = torch.cat([cur_spike, history_spike], dim=0)
            
        return cur_spike, cur_state, data_idx
        
    def __len__(self):
        return len(self.data_idx)

class NeuralFolder(data.Dataset):
    def __init__(self, filename, history):
        self.history = history
        self.mat = loadmat(filename)
        self.spike = np.array(self.mat['trainSpike']).T.tolist()
        self.state = self.mat['trainState']
    
    def __getitem__(self, index):
        index = index % len(self.spike)
        cur_spike = torch.FloatTensor(self.spike[index])
        if math.isnan(self.state[index]):
            cur_state = torch.LongTensor([0]) # 0 - nan
        else:
            cur_state = torch.LongTensor([1]) # 1 - valid

        for i in range(1, self.history):
            history_idx = index - i
            if history_idx < 0:
                history_idx = 0
            history_spike = torch.FloatTensor(self.spike[history_idx])
            cur_spike = torch.cat([cur_spike, history_spike], dim=0)
        
        return cur_spike, cur_state, index

    def __len__(self):
        return len(self.spike)

class TestFolder(data.Dataset):
    def __init__(self, filename, history):
        self.history = history
        self.mat = loadmat(filename)
        self.spike = np.array(self.mat['testSpike']).T.tolist()
    
    def __getitem__(self, index):
        index = index % len(self.spike)
        cur_spike = torch.FloatTensor(self.spike[index])

        for i in range(1, self.history):
            history_idx = index - i
            if history_idx < 0:
                history_idx = 0
            history_spike = torch.FloatTensor(self.spike[history_idx])
            cur_spike = torch.cat([cur_spike, history_spike], dim=0)
        
        return cur_spike, index

    def __len__(self):
        return len(self.spike)
