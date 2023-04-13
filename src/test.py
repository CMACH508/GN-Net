
import torch
import torch.nn.parallel
from torch.utils.data import DataLoader
import torch.nn.functional as F
from model import GNNet
import torch.utils.data as Data
import pickle
from metrics import cal_metrics
import numpy as np


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = GNNet()
model.to(device)
model_path = '../checkpoint/model_0.9146.pth'
model.load_state_dict(torch.load(model_path))
print("load done")

npoints = 400
dataset_test = pickle.load(open('../../dataset/dataset_pc_balanced_test_{}p_norm_rcs.pkl'.format(npoints), 'rb'))

pc_test = torch.tensor(np.array(dataset_test['pc_test']))
norm_test = torch.tensor(np.array(dataset_test['norm_test']))
y_test = torch.tensor(dataset_test['y_test'], dtype=torch.long)
print("pc_test.shape", pc_test.shape)
print("norm_test.shape", norm_test.shape)

dataset_test_cube = pickle.load(open('../../dataset/dataset_cta_balanced_test_gnnet.pkl', 'rb'))

cube_test = torch.tensor(dataset_test_cube['vox_test'])
print("cube_test.shape", cube_test.shape)

test_dataset = Data.TensorDataset(pc_test, norm_test, cube_test, y_test)
test_total = y_test.shape[0]/32
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

y_true = []
y_pred = []
y_prob = []

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
    pred_choice = pred_choice.cpu().detach().numpy()
    cls_idx = cls_idx.cpu().detach().numpy()

    prob = (pred/32).cpu().detach().numpy()
    prob = prob[1]
    y_prob.append(prob)

    y_pred.append(pred_choice)
    y_true.append(cls_idx[0])

    correct_t = 0
    if pred_choice == cls_idx[0]:
        correct_t = 1
    test_correct += correct_t

    test_correct_each[cls_idx[0]] += correct_t
    test_total_each[cls_idx[0]] += 1

test_acc = test_correct / float(test_total)
test_acc_each = [test_correct_each[c]/test_total_each[c] for c in range(2)]

print("test accuracy: {}".format(test_acc))
print("0 accuracy: {}, 1 accuracy: {}".format(test_acc_each[0], test_acc_each[1]))

cal_metrics(y_true, y_pred, y_prob)


import numpy as np
ret = [y_prob, y_pred, y_true]
np.savetxt('../save/test_balanced.txt', ret)

'''
load done
pc_test.shape torch.Size([2624, 400, 3])
norm_test.shape torch.Size([2624, 400, 3])
cube_test.shape torch.Size([2624, 1, 48, 48, 48])
test accuracy: 0.9146341463414634
0 accuracy: 0.926829268292683, 1 accuracy: 0.9024390243902439
[[38  3]
 [ 4 37]]
Accuracy: 91.46
Precision: 92.5
Recall: 90.24
AUC: 94.53
AUPR: 95.35
F1 score: 91.36
'''