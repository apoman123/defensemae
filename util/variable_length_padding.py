import torch
import torch.nn as nn

# please use this function together with your collate function
def padding(spec, in_chanel, embed_dim, patch_size=16, smallest_length=1024):
    # padding_mask: 0->the pads, 1->spectrogram data
    # pad them all to length smallest length
    embeder = nn.Conv2d(in_chanel, embed_dim, kernel_size=(patch_size, patch_size), stride=(patch_size, patch_size), bias=False)
    embeder.weight.requires_grad = False

    # spectrogram dimension is H, W
    H, W = spec.shape
    num_pads = smallest_length - W
    padding_mask = torch.cat([torch.ones(H, W), torch.zeros(H, num_pads)], dim=-1)
    spec = torch.cat([spec, torch.zeros(H, num_pads)], dim=-1)
    return spec, padding_mask
    