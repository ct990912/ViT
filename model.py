import torch
from torch import nn
import math
from torchsummary import summary


class Attention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim

        # 下面两行代码是为了获得能被num_heads整除的all_head_dim，方便划分Q、K、V矩阵
        # 如果embed_dim可以被num_heads除尽，则all_head_dim == embed_dim
        self.head_dim = embed_dim // num_heads
        self.all_head_dim = self.head_dim * num_heads

        # 一步生成Q、K、V矩阵，所以乘以3
        self.qkv = nn.Linear(embed_dim, 3 * self.all_head_dim, bias=False)

        self.softmax = nn.Softmax(-1)
        self.proj = nn.Linear(self.all_head_dim, embed_dim)

    def forward(self, x):
        # 获取qkv矩阵，将qkv拆分成Q、K、V矩阵
        qkv = self.qkv(x)  # batch_size * (n_patches+1) * embed_dim * (3*self.all_head_dim)
        Q, K, V = torch.chunk(qkv, 3, -1)  # batch_size * (n_patches+1) * embed_dim * self.all_head_dim

        # batch_size * num_heads * (n_patches+1) * head_dim
        Q = Q.view(Q.size(0), Q.size(1), self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(K.size(0), K.size(1), self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(V.size(0), V.size(1), self.num_heads, self.head_dim).transpose(1, 2)

        # 以下是multi-head-attention操作
        atten = self.softmax(torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(
            self.head_dim))  # batch_size * num_heads * (n_patches+1) * (n_patches+1)
        out = torch.matmul(atten, V)  # batch_size * num_heads * (n_patches+1) * head_dim
        out = out.transpose(1, 2).contiguous().flatten(2)  # batch_size * (n_patches+1) * all_head_dim
        out = self.proj(out)  # batch_size * (n_patches+1) * embed_dim
        return out


class MLP(nn.Module):
    def __init__(self, embed_dim, mlp_ratio=6, dropout=0):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(embed_dim, int(embed_dim * mlp_ratio))
        self.fc2 = nn.Linear(int(embed_dim * mlp_ratio), embed_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class PatchEmbedding(nn.Module):
    def __init__(self, image_size, patch_size, in_channels, embed_dim):
        super(PatchEmbedding, self).__init__()
        n_patches = (image_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, patch_size, patch_size, bias=False)

        # class_token 和 position_embedding 都是可以学习的参数
        self.class_token = nn.Parameter(torch.zeros((1, 1, embed_dim)))
        self.position_embedding = nn.Parameter(torch.randn((1, n_patches + 1, embed_dim)))

    def forward(self, x):
        # class_token：在训练中将class_token与patch embedding做Attention，最终输出的class_token就包含了分类信息
        cls_tokens = self.class_token.expand(x.size(0), -1, -1)  # batch_size * 1 * embedding_dim

        x = self.patch_embed(x)  # batch_size * embedding_dim * (image_size // patch_size) * (image_size // patch_size)
        x = x.flatten(2)  # batch_size * embedding_dim * n_patches
        x = x.transpose(2, 1)  # batch_size * n_patches * embedding_dim

        # 将[class] embedding与patch_embedding拼接
        x = torch.concat([cls_tokens, x], dim=1)  # batch_size * (n_patches+1) * embedding_dim

        # 加入Position Embedding
        posit = self.position_embedding.repeat(x.size(0), 1, 1)
        x = x + posit  # batch_size * (n_patches+1) * embedding_dim
        return x


class EncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(EncoderLayer, self).__init__()
        self.attn = Attention(embed_dim, num_heads=num_heads)
        self.attn_norm = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim)
        self.mlp_norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # 维度都是 batch_size * (num_of_patch+1) * embedding_dim
        h = x
        x = self.attn_norm(x)
        x = self.attn(x)
        x = h + x
        x = self.mlp_norm(x)
        x = self.mlp(x)
        x = h + x
        return x


class Encoder(nn.Module):
    def __init__(self, embed_dim, num_heads, depth):
        super(Encoder, self).__init__()
        layer_list = []
        for i in range(depth):
            layer_list.append(EncoderLayer(embed_dim, num_heads))
        self.layers = nn.Sequential(*layer_list)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):  # batch_size * (num_of_patch+1) * embedding_dim
        x = self.layers(x)  # batch_size * (num_of_patch+1) * embedding_dim
        return self.norm(x)  # batch_size * (num_of_patch+1) * embedding_dim


class ViT(nn.Module):
    def __init__(self, image_size, patch_size=16, in_channel=3, num_classes=10, embed_dim=768, depth=3, num_heads=8,
                 mlp_ratio=4):
        super(ViT, self).__init__()

        self.patch_embed = PatchEmbedding(image_size, patch_size, in_channel, embed_dim)
        self.encoders = Encoder(embed_dim, num_heads, depth)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):  # batch_size * channel * W * H

        # 将输入的图片变成Patch Embedding + Position Embedding + [class] embedding
        x = self.patch_embed(x)  # batch_size * (num_of_patch+1) * embedding_dim

        x = self.encoders(x)  # batch_size * (num_of_patch+1) * embedding_dim

        # 将[class] embedding单独取出来输入到分类器中得出分类分数
        x = x[:, 0, :]  # batch_size * embedding_dim
        x = self.classifier(x)  # batch_size * num_classes
        return x


if __name__ == "__main__":

    net = ViT(224, 4, 3, 10, 128, 3, 8, 4)
    summary(net, (3, 224, 225), 1, 'cpu')
