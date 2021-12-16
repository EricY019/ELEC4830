import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import os
from scipy import io

from data import ValidFolder
from data import TestFolder
from model import LinearModel

args = {
    'max_epoch': 30,
    'learning_rate': 1e-2,
    'warm_up_epoch': 3,
    'margin': 0.5, # nan type confidence, default 0.5
    'history': 8, # num of current and history inputs
    'print_per_iter': 5,
    'seed': 4830,
}
# fix random seed
np.random.seed(args['seed'])
torch.manual_seed(args['seed'])
# dataset and dataloader
valid_set = ValidFolder('ELEC4830_Final_project.mat', args['history'])
test_set = TestFolder('ELEC4830_Final_project.mat', args['history'])
valid_loader = DataLoader(valid_set, batch_size=1, num_workers=0, shuffle=False)
test_loader = DataLoader(test_set, batch_size=1, num_workers=0, shuffle=False)
# BCE-loss
BCEloss = nn.CrossEntropyLoss().type(torch.FloatTensor)
# model
model = LinearModel()
# path
path = r'D:\EricYANG\HKUST\21Fall\elec4830\project'
# prediction
pred_all = []
majority = []

def main():
    for k in range(1, 6):
        pred = []
        save_path = os.path.join(path, 'savemodel', f'{k}.pth')
        model.load_state_dict(torch.load(save_path))
        with torch.no_grad():
            model.eval()

        for _, sample in enumerate(tqdm(test_loader)):
            spike = sample[0]
            idx = int(sample[1])

            score = model(spike)
            ind = int(torch.argmax(score))
            pred.append((idx, ind))
        
        pred.sort(key=lambda tup: tup[0])
        pred_all.append(pred)

    for idx in range(len(pred_all[0])):
        sum = 0
        for j in range(len(pred_all)):
            sum += pred_all[j][idx][1]
        if sum > 2:
            majority.append(1)
        elif sum <= 2:
            majority.append(0)
    
    # io.savemat('result_nan.mat', {"testStatenan": majority})
    io.savemat('result.mat', {"testState": majority})
    # true_1 = 0
    # false_1 = 0
    # false_0 = 0
    # true_0 = 0
    # confusion = np.zeros([2, 2])
    # for _, sample in enumerate(valid_loader):
    #     state = int(torch.squeeze(sample[1]))
    #     idx = int(sample[2])
    #     predict_state = [sub[1] for sub in majority if idx == sub[0]]
    #     if predict_state[0] == 0 and state == 0:
    #         true_0 += 1
    #     elif predict_state[0] == 1 and state == 1:
    #         true_1 += 1
    #     elif predict_state[0] == 0 and state == 1:
    #         false_0 += 1
    #     elif predict_state[0] == 1 and state == 0:
    #         false_1 += 1
    #     confusion[state][predict_state[0]] += 1
    # acc = (true_1 + true_0) / (true_1 + true_0 + false_1 + false_0)
    # print("acc:", acc)
    # print("Confusion:", confusion)        
        

if __name__ == '__main__':
    main()