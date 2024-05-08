import torch
import torch.nn as nn

def padding(spec_batch, in_chanel, embed_dim, patch_size=16, smallest_length=1024):
    # the default longest length of a spectrogram is 1024
    padded_specs = torch.tensor([])
    embeder = nn.Conv2d(in_chanel, embed_dim, kernel_size=(patch_size, patch_size), stride=(patch_size, patch_size), bias=False)
    embeder.weight.requires_grad = False
    N, C, H, W = spec_batch.shape
    longest = smallest_length

    # find the longest 
    for idx in range(N):
        spec = spec_batch[idx, :, :, :]
        spec_n, spec_c, spec_h, spec_w = spec.shape
        if spec_w > longest:
            longest = spec_w

    # pad the spectrogram
    for idx in range(N):
        spec = spec_batch[idx, :, :, :]
        spec_n, spec_c, spec_h, spec_w = spec.shape
        if spec_w < longest:
            pads = torch.zeros(spec_n, spec_c, spec_h, longest - spec_w)
            padded_spec = torch.cat([spec, pads], dim=-1)
            padded_specs = torch.cat([padded_specs, padded_spec], dim=0)

    # get the padding mask
    padding_masks = embeder(padded_specs).flatten(2).transpose(1, 2)
    padding_masks = torch.where(padding_masks == 0, 1, 0).bool()

    return padded_specs, padding_masks