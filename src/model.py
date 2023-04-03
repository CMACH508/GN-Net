from branch1 import Net_b1
from branch2 import Net_b2
import torch
import torch.nn as nn


class GNNet(nn.Module):
    def __init__(self, n_classes=2, input_shape=(48, 48, 48), dim = 256, depth = 6, heads = 16, dim_head = 64, mlp_dim = 1024):
        super(GNNet, self).__init__()
        self.branch1 = Net_b1()
        self.branch2 = Net_b2()

        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(64, n_classes)
        )

    def forward(self, pos, norm, batch, cube):
        x1 = self.branch1(pos, norm, batch)
        x2 = self.branch2(cube)
        x = torch.cat([x1, x2], dim=-1)
        x = self.fc(x)

        return x


if __name__ == "__main__":
    net = GNNet()
    pos = torch.rand((32, 400, 3))
    norm = torch.rand((32, 400, 3))
    B, N = pos.shape[0], pos.shape[1]
    pos = pos.view(B * N, -1)
    norm = norm.view(B * N, -1)
    batch = []
    for i in range(B):
        for j in range(N):
            batch.append(i)
    batch = torch.tensor(batch)
    cube = torch.rand([32, 1, 48, 48, 48])
    pred = net(pos, norm, batch, cube)
    print(pred)
