import numpy as np
import torch
from torch.utils.data import default_collate


def slice(x, section_length=10, overlap=5):
    start_indices = np.arange(0, len(x) - overlap, section_length - overlap)
    sections = [x[start : start + section_length] for start in start_indices]

    # If the last section is not of length 'section_length', you can decide whether to keep or discard it
    if len(sections[-1]) < section_length:
        sections.pop(-1)  # Discard the last section

    return np.concatenate(sections, 1).T


def preprocess(samples):
    out = []

    for x in samples["spectrum"]:
        x = np.array(x)
        std, mean = x.std(), x.mean()
        # skipping samples that are all zero
        if std == 0:
            continue
        x = (x - mean) / std
        x = slice(x, 194, 97)
        x = np.pad(x, pad_width=((1, 0), (2, 0)), mode="constant", constant_values=0)

        x[0, 0] = (mean - 2) / 2
        x[0, 1] = (std - 2) / 8

        out.append(x)
    # print(len(out))
    return torch.tensor(np.array(out))


# for training we drop chunks of the spectrum
def drop_chunks(batch, size=5):
    batch = batch.clone()
    # random start location between 0 and length of the spectrum
    start1 = torch.randint(0, batch.shape[1] - 3 * size - 2, (1,)).item()
    start2 = torch.randint(
        start1 + size + 1, batch.shape[1] - 2 * size - 1, (1,)
    ).item()
    start3 = torch.randint(start2 + size + 1, batch.shape[1] - size, (1,)).item()
    batch[:, start1 : start1 + size] *= 0
    batch[:, start2 : start2 + size] *= 0
    batch[:, start3 : start3 + size] *= 0
    return batch


def spectrum_collate_fn(sample):
    sample = default_collate(sample)
    preprocessed_sample = preprocess(sample)
    return {"input": drop_chunks(preprocessed_sample), "target": preprocessed_sample}
