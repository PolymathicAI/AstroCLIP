import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence


class SpectrumCollator:
    def __init__(self, num_chunks: int = 6, chunk_width: int = 50):
        self.num_chunks = num_chunks
        self.chunk_width = chunk_width

    def __call__(self, samples, mlm=None):
        preprocessed_samples = [
            preprocess(el["spectrum"])
            for el in samples
            if np.array(el["spectrum"]).std() > 0
        ]
        out = {
            "input": pad_sequence(
                [
                    mask_seq(torch.tensor(el), self.num_chunks, self.chunk_width)
                    for el in preprocessed_samples
                ],
                batch_first=True,
                padding_value=0,
            ),
            "target": pad_sequence(
                [torch.tensor(el) for el in preprocessed_samples],
                batch_first=True,
                padding_value=0,
            ),
        }

        return out


def preprocess(x):
    x = np.array(x)
    std, mean = x.std(), x.mean()
    # skipping samples that are all zero
    if std != 0:
        x = (x - mean) / std
        x = slice(x, 20, 10)
        x = np.pad(x, pad_width=((1, 0), (2, 0)), mode="constant", constant_values=0)

        x[0, 0] = (mean - 2) / 2
        x[0, 1] = (std - 2) / 8

        return x
    else:
        return None


def mask_seq(seq, num_chunks, chunk_width):
    # randomly masking contiguous sections of the sequence
    # making sure the separation between chunks is at least chunk_width

    len_ = seq.shape[0]
    num_chunks = num_chunks
    chunk_width = chunk_width

    # Ensure there's enough space for the chunks and separations
    total_width_needed = num_chunks * chunk_width + (num_chunks - 1) * chunk_width
    if total_width_needed > len_:
        raise ValueError("Sequence is too short to mask")

    masked_seq = seq.clone()

    for i in range(num_chunks):
        start = (i * len_) // num_chunks
        loc = torch.randint(0, len_ // num_chunks - chunk_width, (1,)).item()
        masked_seq[loc + start : loc + start + chunk_width] = 0

    return masked_seq


def slice(x, section_length: int = 10, overlap: int = 5):
    start_indices = np.arange(0, len(x) - overlap, section_length - overlap)
    sections = [x[start : start + section_length] for start in start_indices]

    # If the last section is not of length 'section_length', you can decide whether to keep or discard it
    if len(sections[-1]) < section_length:
        sections.pop(-1)  # Discard the last section

    return np.concatenate(sections, 1).T
