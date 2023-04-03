import torch
import torch.nn as nn
from einops import rearrange, repeat
from transformer import Transformer


class Net_b2(nn.Module):
    def __init__(self, n_classes=2, input_shape=(48, 48, 48), dim = 256, depth = 6, heads = 16, dim_head = 64, mlp_dim = 1024):
        super(Net_b2, self).__init__()
        self.n_classes = n_classes
        self.input_shape = input_shape
        self.ConvBlock1 = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=32, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool3d(2)
        )

        # linear projection for patches
        self.patch_embeddings = nn.Conv3d(in_channels=32, out_channels=dim, kernel_size=2, stride=2)


        num_patches = 125
        emb_dropout = 0.1
        dropout = 0.1
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

    def forward(self, x):
        x = self.ConvBlock1(x)

        x = self.patch_embeddings(x)
        # print(x.shape)
        x = torch.flatten(x, start_dim=2)  # batch_size dim n_patch
        x = rearrange(x, 'b d n -> b n d')  # batch_size n_patch dim

        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '1 n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        x = self.transformer(x)
        x = x[:, 0]

        return x


if __name__ == "__main__":
    net = Net_b2()
    data = torch.rand([8, 4, 48, 48, 48])
    pred = net(data)
    print(pred)
