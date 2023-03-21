import torch
import torch.nn.functional as F


def avg_ch_pool(features):
    return F.adaptive_avg_pool2d(features, (1, 1)).squeeze(2).squeeze(2)


def max_ch_pool(features):
    return F.adaptive_max_pool2d(features, (1, 1)).squeeze(2).squeeze(2)


def add_avg_max_ch_pool(features):
    return avg_ch_pool(features) + max_ch_pool(features)


def cat_avg_max_ch_pool(features):
    return torch.cat((avg_ch_pool(features), max_ch_pool(features)), dim=1)


def avg_time_pool(features):
    return torch.mean(features, dim=3).view(features.size(0), -1)


def avg_max_time_pool(features):
    return torch.mean(features, dim=3).view(features.size(0), -1) + \
           torch.amax(features, dim=3).view(features.size(0), -1)

