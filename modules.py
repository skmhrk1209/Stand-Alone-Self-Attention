import torch
from torch import nn


def output_shape(inputs, kernel_size, padding, stride, dilation):
    return [(size + padding * 2 - dilation * (kernel_size - 1) - 1) // stride + 1 for size in inputs.shape[-2:]]


class SelfAttention(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1, dilation=1, groups=1, bias=False):

        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.groups = groups

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, groups=groups, bias=bias)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=1, groups=groups, bias=bias)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, groups=groups, bias=bias)

        self.row_embeddings = nn.Parameter(torch.randn(out_channels // 2, kernel_size))
        self.col_embeddings = nn.Parameter(torch.randn(out_channels // 2, kernel_size))

        self.unfold1 = nn.Unfold(kernel_size=1, stride=stride)
        self.unfold2 = nn.Unfold(kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation)
        self.unfold3 = nn.Unfold(kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation)

    def forward(self, inputs):

        queries = self.conv1(inputs)
        queries = self.unfold1(queries)
        queries = queries.reshape(queries.size(0), self.groups, self.out_channels // self.groups, -1, queries.size(-1))
        queries = queries.permute(0, 4, 1, 2, 3) # Query: [B, N, G, C // G, 1]

        keys = self.conv2(inputs)
        keys = self.unfold2(keys)
        keys = keys.reshape(keys.size(0), self.groups, self.out_channels // self.groups, -1, keys.size(-1))
        keys = keys.permute(0, 4, 1, 2, 3) # Key: [B, N, G, C // G, K^2]

        row_embeddings = self.row_embeddings.unsqueeze(-1).expand(-1, -1, self.kernel_size)
        col_embeddings = self.col_embeddings.unsqueeze(-2).expand(-1, self.kernel_size, -1)
        embeddings = torch.cat((row_embeddings, col_embeddings), dim=0)
        embeddings = embeddings.reshape(self.groups, self.out_channels // self.groups, -1)
        embeddings = embeddings.unsqueeze(0).unsqueeze(1) # Embedding: [1, 1, G, C // G, K^2]

        attentions = torch.matmul(torch.transpose(queries, -2, -1), keys + embeddings)
        attentions = nn.functional.softmax(attentions, dim=-1) # Attention: [B, N, G, 1, K^2]

        values = self.conv3(inputs)
        values = self.unfold3(values)
        values = values.reshape(values.size(0), self.groups, self.out_channels // self.groups, -1, values.size(-1))
        values = values.permute(0, 4, 1, 2, 3) # Value: [B, N, G, C // G, K^2]

        outputs = torch.matmul(values, torch.transpose(attentions, -2, -1)) # Self-Attention: [B, N, G, C // G, 1]
        outputs = outputs.permute(0, 2, 3, 4, 1)
        outputs = outputs.reshape(outputs.size(0), self.out_channels, *output_shape(inputs, self.kernel_size, self.padding, self.stride, self.dilation))

        return outputs


class AttentionStem(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1, dilation=1, groups=1, mixtures=1, bias=False):

        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.mixtures = mixtures

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, groups=groups, bias=bias)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=1, groups=groups, bias=bias)
        self.conv3 = nn.Conv2d(in_channels, out_channels * mixtures, kernel_size=1, groups=groups, bias=bias)

        self.row_embeddings = nn.Parameter(torch.randn(out_channels, kernel_size))
        self.col_embeddings = nn.Parameter(torch.randn(out_channels, kernel_size))
        self.mix_embeddings = nn.Parameter(torch.randn(out_channels, mixtures))

        self.unfold1 = nn.Unfold(kernel_size=1, stride=stride)
        self.unfold2 = nn.Unfold(kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation)
        self.unfold3 = nn.Unfold(kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation)

    def forward(self, inputs):

        queries = self.conv1(inputs)
        queries = self.unfold1(queries)
        queries = queries.reshape(queries.size(0), self.groups, self.out_channels // self.groups, -1, queries.size(-1))
        queries = queries.permute(0, 4, 1, 2, 3) # Query: [B, N, G, C // G, 1]

        keys = self.conv2(inputs)
        keys = self.unfold2(keys)
        keys = keys.reshape(keys.size(0), self.groups, self.out_channels // self.groups, -1, keys.size(-1))
        keys = keys.permute(0, 4, 1, 2, 3) # Key: [B, N, G, C // G, K^2]

        attentions = torch.matmul(torch.transpose(queries, -2, -1), keys)
        attentions = nn.functional.softmax(attentions, dim=-1) # Attention: [B, N, G, 1, K^2]

        row_embeddings = self.row_embeddings.unsqueeze(1).unsqueeze(3)
        col_embeddings = self.col_embeddings.unsqueeze(1).unsqueeze(2)
        mix_embeddings = self.mix_embeddings.unsqueeze(2).unsqueeze(3)

        weights = torch.sum((row_embeddings + col_embeddings) * mix_embeddings, dim=0)
        weights = nn.functional.softmax(weights, dim=0)
        weights = weights.reshape(weights.size(0), -1)
        weights = weights.unsqueeze(0).unsqueeze(1).unsqueeze(2).unsqueeze(3) # Weight: [1, 1, 1, 1, M, K^2]

        values = self.conv3(inputs)
        values = self.unfold3(values)
        values = values.reshape(values.size(0), self.groups, self.out_channels // self.groups, self.mixtures, -1, values.size(-1))
        values = values.permute(0, 5, 1, 2, 3, 4) # Values: [B, N, G, C // G, M, K^2]
        values = torch.sum(values * weights, dim=-2) # Values: [B, N, G, C // G, K^2]
     
        outputs = torch.matmul(values, torch.transpose(attentions, -2, -1))
        outputs = outputs.permute(0, 2, 3, 4, 1) # Self-Attention: [B, G, C // G, 1, N]
        outputs = outputs.reshape(outputs.size(0), self.out_channels, *output_shape(inputs, self.kernel_size, self.padding, self.stride, self.dilation))

        return outputs