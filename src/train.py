import sys
sys.path.append("..")
import random
import torch
import torch.nn.parallel
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from model import GNNet
import torch.utils.data as Data
import pickle
import numpy as np
from torch.nn import DataParallel
import os
import argparse

# parser = argparse.ArgumentParser(description='Train neural net on .types data.')
# parser.add_argument('-g', '--gpu', type=str, default='0,1')
# args = parser.parse_args()
# os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

def train(n_epoch = 200, BS = 128, LR=1e-4, step_size=200):
    manualSeed = random.randint(1, 10000)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Loading data...")
    npoints = 400
    dataset_train = pickle.load(open('../../dataset/dataset_pc_balanced_train_{}p_norm_rcs.pkl'.format(npoints), 'rb'))
    dataset_test = pickle.load(open('../../dataset/dataset_pc_balanced_test_{}p_norm_rcs.pkl'.format(npoints), 'rb'))

    pc_train = torch.tensor(np.array(dataset_train['pc_train']))
    pc_test = torch.tensor(np.array(dataset_test['pc_test']))
    norm_train = torch.tensor(np.array(dataset_train['norm_train']))
    norm_test = torch.tensor(np.array(dataset_test['norm_test']))
    y_train = torch.tensor(dataset_train['y_train'], dtype=torch.long)
    y_test = torch.tensor(dataset_test['y_test'], dtype=torch.long)
    print("pc_train.shape", pc_train.shape)
    print("pc_test.shape", pc_test.shape)
    print("norm_train.shape", norm_train.shape)
    print("norm_test.shape", norm_test.shape)

    dataset_train_cube = pickle.load(open('../../dataset/dataset_cta_balanced_train.pkl', 'rb'))
    dataset_test_cube = pickle.load(open('../../dataset/dataset_cta_balanced_test.pkl', 'rb'))

    cube_train = torch.tensor(dataset_train_cube['vox_train'])
    cube_test = torch.tensor(dataset_test_cube['vox_test'])
    cube_train = cube_train[:,:1,:,:,:]
    cube_test = cube_test[:,:1,:,:,:]
    print("cube_train.shape", cube_train.shape)
    print("cube_test.shape", cube_test.shape)

    train_dataset = Data.TensorDataset(pc_train, norm_train, cube_train, y_train)
    test_dataset = Data.TensorDataset(pc_test, norm_test, cube_test, y_test)
    train_total = y_train.shape[0]
    test_total = y_test.shape[0]/32
    train_dataloader = DataLoader(train_dataset, batch_size=BS, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    num_class = 2
    model = GNNet(num_class)
    # model = DataParallel(model)

    train_acc_list = []
    test_acc_list = []
    train_acc_each_list = []
    test_acc_each_list = []

    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.5)
    model.to(device)

    best_acc = 0

    for epoch in range(n_epoch):
        print("epoch: {} lr: {}".format(epoch, optimizer.param_groups[0]['lr']))
        train_correct = 0
        train_correct_each = [0 for c in range(2)]
        train_total_each = [0 for c in range(2)]

        for i, (pos, norm, cube, cls_idx) in enumerate(train_dataloader, 0):
            pos, norm, cube, cls_idx = pos.to(device), norm.to(device), cube.to(device), cls_idx.to(device)
            pos = pos.float()
            norm = norm.float()
            cube = cube.float()

            B, N = pos.shape[0], pos.shape[1]
            pos = pos.view(B*N, -1)
            norm = norm.view(B*N, -1)
            batch = []
            for i in range(B):
                for j in range(N):
                    batch.append(i)
            batch = torch.tensor(batch).to(device)

            optimizer.zero_grad()
            model = model.train()
            pred = model(pos, norm, batch, cube)
            loss = F.cross_entropy(pred, cls_idx)
            loss.backward()
            optimizer.step()

            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(cls_idx.data).cpu().sum()
            train_correct += correct

            for number in range(2):
                train_correct_each[number] += torch.sum(pred_choice[cls_idx == number] == number).item()
                train_total_each[number] += cls_idx[cls_idx == number].shape[0]

        train_acc = train_correct / float(train_total)
        train_acc_list.append(train_acc)

        train_acc_each = [train_correct_each[c]/train_total_each[c] for c in range(2)]
        train_acc_each_list.append(train_acc_each)

        print("train acc:{:.4f}, [0 acc:{:.4f}, 1 acc:{:.4f}]".format(train_acc, train_acc_each[0], train_acc_each[1]))
        print("loss: {}".format(loss.item()))
        scheduler.step()

        test_correct = 0
        test_correct_each = [0 for c in range(2)]
        test_total_each = [0 for c in range(2)]

        for i, (pos, norm, cube, cls_idx) in enumerate(test_dataloader, 0):
            pos, norm, cube, cls_idx = pos.to(device), norm.to(device), cube.to(device), cls_idx.to(device)
            pos = pos.float()
            norm = norm.float()
            cube = cube.float()

            B, N = pos.shape[0], pos.shape[1]
            pos = pos.view(B*N, -1)
            norm = norm.view(B*N, -1)
            batch = []
            for i in range(B):
                for j in range(N):
                    batch.append(i)
            batch = torch.tensor(batch).to(device)
            
            model = model.eval()
            pred = model(pos, norm, batch, cube)
            pred = F.softmax(pred, dim=1)
            pred = pred.sum(dim=0)
            pred_choice = pred.data.max(0)[1]

            correct_t = 0
            if pred_choice == cls_idx[0]:
                correct_t = 1
            test_correct += correct_t

            test_correct_each[cls_idx[0]] += correct_t
            test_total_each[cls_idx[0]] += 1

        test_acc = test_correct / float(test_total)
        test_acc_list.append(test_acc)
        test_acc_each = [test_correct_each[c]/test_total_each[c] for c in range(2)]
        test_acc_each_list.append(test_acc_each)

        print("test acc:{:.4f}, [0 acc:{:.4f}, 1 acc:{:.4f}]".format(test_acc, test_acc_each[0], test_acc_each[1]))

        if test_acc > best_acc:
            print('Saving..')
            torch.save(model.state_dict(), '../checkpoint/model_{:.4f}.pth'.format(test_acc))
            best_acc = test_acc

        print("best test accuracy: {}".format(best_acc))


if __name__ == "__main__":
    n_epoch = 200
    BS = 32
    LR = 1e-4
    step_size = 200
    train(n_epoch, BS, LR, step_size)
