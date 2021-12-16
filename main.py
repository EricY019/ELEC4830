import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from tqdm import tqdm
import numpy as np
import os
import math
import time

from data import ValidFolder
from data import NanFolder
from data import NeuralFolder
from model import LinearModel

args = {
        'max_epoch': 30,
        'learning_rate': 1e-2,
        'batch_size': 100,
        'warm_up_epoch': 3,
        'margin': 0.4, # nan type confidence, default 0.5
        'history': 8, # num of current and history inputs
        'print_per_iter': 5,
        'seed': 4830,
        'log_suffix': 'final'
    }
# fix random seed
np.random.seed(args['seed'])
torch.manual_seed(args['seed'])
# dataset
valid_set = ValidFolder('ELEC4830_Final_project.mat', args['history'])
nan_set = NanFolder('ELEC4830_Final_project.mat', args['history'])
set = NeuralFolder('ELEC4830_Final_project.mat', args['history'])
# BCE-loss
BCEloss = nn.CrossEntropyLoss().type(torch.FloatTensor)
# Softmax
softmax = nn.Softmax()
# cross-validation
kf = KFold(n_splits=5, shuffle=False)
# local path
path = r'D:\EricYANG\HKUST\21Fall\elec4830\project'
log_path = os.path.join(path, 'log', time.strftime("%Y%m%d-%H%M%S-") + args['log_suffix'] + '.txt')

def main():
    trian_loss_all = []
    val_loss_all = []
    acc_all = []
    confusion_all = []
    # nan/valid prediction
    # k = 1
    # train_set = nan_set
    # val_set = set
    # train_loader = DataLoader(train_set, batch_size=args['batch_size'], num_workers=0, shuffle=False)
    # val_loader = DataLoader(val_set, batch_size=args['batch_size'], num_workers=0, shuffle=False)
    # model = LinearModel().type(torch.FloatTensor).train()
    # optimizer = optim.Adam(model.parameters(), lr=args['learning_rate'], betas=(0.9, 0.99), eps=6e-8, weight_decay=5e-4)
    # warm_up_with_cosine_lr = lambda epoch: epoch / args['warm_up_epoch'] if epoch <= args['warm_up_epoch'] else 0.5 * \
    #                      (math.cos((epoch-args['warm_up_epoch'])/(args['max_epoch']-args['warm_up_epoch'])*math.pi)+1)
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warm_up_with_cosine_lr)
    # model, train_loss, val_loss, acc, confusion = train(model, optimizer, scheduler, train_loader, val_loader, 1)
    # trian_loss_all.append(train_loss)
    # val_loss_all.append(val_loss)
    # acc_all.append(acc)
    # confusion_all.append(confusion)
    # for i in range(len(model)):
    #     save_path = os.path.join(path, 'savemodel', 'nan'f'{i}.pth')
    #     print("Saving model to", save_path)
    #     torch.save(model[i].state_dict(), save_path)
    #####
    # 0/1 prediction
    k = 0
    for train_index, val_index in kf.split(valid_set):
        k += 1
        train_set = Subset(valid_set, train_index)
        val_set = Subset(valid_set, val_index)
        train_loader = DataLoader(train_set, batch_size=args['batch_size'], num_workers=0, shuffle=False)
        val_loader = DataLoader(val_set, batch_size=1, num_workers=0, shuffle=False)
        
        model = LinearModel().type(torch.FloatTensor).train()
        # optimizer = optim.Adam(model.parameters(), lr=args['learning_rate'], betas=(0.9, 0.99), eps=6e-8, weight_decay=5e-4)
        # optimizer = optim.SGD(model.parameters(), lr=args['learning_rate'], momentum=0.9)
        optimizer = optim.RMSprop(model.parameters(), lr=args['learning_rate'])
        warm_up_with_cosine_lr = lambda epoch: epoch / args['warm_up_epoch'] if epoch <= args['warm_up_epoch'] else 0.5 * \
                             (math.cos((epoch-args['warm_up_epoch'])/(args['max_epoch']-args['warm_up_epoch'])*math.pi)+1)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warm_up_with_cosine_lr)
        model, train_loss, val_loss, acc, confusion = train(model, optimizer, scheduler, train_loader, val_loader, k)

        trian_loss_all.append(train_loss)
        val_loss_all.append(val_loss)
        acc_all.append(acc)
        confusion_all.append(confusion)

        # save_path = os.path.join(path, 'savemodel', f'{k}.pth')
        # print("Saving model to", save_path)
        # torch.save(model.state_dict(), save_path)
    #####
    print("Confusion", confusion_all)
    for epch in range(args['max_epoch']):
        train_loss, val_loss, acc = 0, 0, 0
        for fold in range(k):
            train_loss += trian_loss_all[fold][epch][1]
            val_loss += val_loss_all[fold][epch][1]
            acc += acc_all[fold][epch][1]
        train_loss /= k
        val_loss /= k
        acc /= k
        # log = "curr_epoch: %d, train-loss: %5f, val-loss: %5f, acc: %5f\n"%\
        #         (epch + 1, train_loss, val_loss, acc)
        # open(log_path, 'a').write(log)


def train(model, optimizer, scheduler, train_loader, val_loader, k):
    curr_epoch = 1
    curr_iter = 1
    loss = None
    train_loss = []
    val_loss = []
    accuracy = []
    best_model, best_acc = None, 0 # top accuracy or top-3 accuracy
    while True:

        print('=====>Training for Epoch', curr_epoch, '<======')  
        for _, sample in enumerate(tqdm(train_loader)):
            spike = sample[0]
            state = torch.squeeze(sample[1])
            score = model(spike)
            
            loss = BCEloss(score, state)
            # L1 regularization
            # loss_lambda = 1e-4
            # loss_norm = sum(p.abs().sum()
            #         for p in model.parameters())
            # loss = loss + loss_lambda * loss_norm

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # if curr_iter % args['print_per_iter'] == 0:
            #     train_loss.append(loss.item())
            #     print("Iteration", curr_iter, "Loss", loss.item())
            #     log = "curr_iter: %d, train-loss: %5f\n"%\
            #     (curr_iter, loss.item())
            #     open(log_path, 'a').write(log) # write log
        
            curr_iter += 1

        if curr_epoch % 1 == 0:
            train_loss.append((curr_epoch, loss.item()))
            model.eval()
            with torch.no_grad():
                confusion, acc, loss = val(model, val_loader, k)
                accuracy.append((curr_epoch, acc))
                val_loss.append((curr_epoch, loss))
        # best acc
        if acc > best_acc:
            best_model = model
        # top three acc
        # for i in range(len(best_acc)):
        #     if acc > best_acc[i]:
        #         best_model[i] = model

        if curr_epoch >= args['max_epoch']:
            return best_model, train_loss, val_loss, accuracy, confusion
        
        curr_epoch += 1
        scheduler.step()
        model.train()


def val(model, val_loader, k):
    loss = 0
    true_1 = 0
    false_1 = 0
    false_0 = 0
    true_0 = 0
    confusion = np.zeros([2, 2])

    print('=====>Validating<======')
    for _, sample in enumerate(tqdm(val_loader)):
        spike = sample[0]
        state = torch.squeeze(sample[1], 0) # 0/1 prediction
        # state = torch.squeeze(sample[1]) # nan/valid prediction
        score = model(spike)
        ## 0/1 prediction ##############
        ind = torch.argmax(score)
        ################################
        ##### nan/valid prediction #####
        # prob = softmax(score) # softmax, probability
        # ind = torch.zeros_like(state).type(torch.LongTensor)
        # for i in range(prob.size(dim=0)):
        #     if prob[i][0] >= args['margin']: # predict as nan only if very high confidence
        #         ind[i] = 0
        #     else:
        #         ind[i] = 1
        ################################
        loss += BCEloss(score, state).item()
        # L1 regularization
        # loss_lambda = 1e-4
        # loss_norm = sum(p.abs().sum()
        #         for p in model.parameters())
        # loss = loss + loss_lambda * loss_norm

        ## 0/1 prediction ##############
        if state.item() == 0 and ind.item() == 0:
            true_0 += 1
        elif state.item() == 1 and ind.item() == 1:
            true_1 += 1
        elif state.item() == 0 and ind.item() == 1:
            false_1 += 1
        elif state.item() == 1 and ind.item() == 0:
            false_0 += 1
        confusion[state.item()][ind.item()] += 1
        ################################
        ## nan/valid prediction ########
        # for i in range(ind.size(dim=0)):
        #     if state[i].item() == 0 and ind[i].item() == 0:
        #         true_0 += 1
        #     elif state[i].item() == 1 and ind[i].item() == 1:
        #         true_1 += 1
        #     elif state[i].item() == 0 and ind[i].item() == 1:
        #         false_1 += 1
        #     elif state[i].item() == 1 and ind[i].item() == 0:
        #         false_0 += 1
        #     confusion[state[i].item()][ind[i].item()] += 1
        ################################
    loss = loss / len(val_loader)
    correct = true_0 + true_1
    acc = correct / len(val_loader) * 100 # 0/1 prediction
    # acc = correct / len(set) * 100 # nan/valid prediction
    sensitivity = true_1 / (true_1 + false_0)
    specificity = true_0 / (true_0 + false_1)
    g_mean = math.sqrt(sensitivity * specificity)

    print("acc:", acc)
    print("g-mean:", g_mean)
    return confusion, acc, loss


if __name__ == '__main__':
    main()