import torch
import glob
from pathlib import Path


def _load_data_shard(file):
    header = torch.from_file(str(file), False, 256, dtype=torch.int32) # header is 256 int32
    assert header[0] == 20240520, "magic number mismatch in the data .bin file"
    assert header[1] == 1, "unsupported version"
    num_tokens = int(header[2]) # number of tokens (claimed)
    with file.open("rb", buffering=0) as f:
        pin_memory = torch.cuda.is_available()
        tokens = torch.empty(num_tokens, dtype=torch.uint16, pin_memory=pin_memory) # avoid pin_memory copy by @YouJiacheng
        f.seek(256 * 4)
        nbytes = f.readinto(tokens.numpy()) # avoid bytes->array copy by @YouJiacheng
        assert nbytes == 2 * num_tokens, "number of tokens read does not match header"
    return tokens


def distributed_data_generator(filename_pattern, batch_size, rank=0, world_size=1):
    files = [Path(file) for file in sorted(glob.glob(filename_pattern))]
    assert batch_size % world_size == 0
    local_batch_size = batch_size // world_size
    file_iter = iter(files) # use itertools.cycle(files) instead if you want to do multi-epoch training
    tokens, pos = _load_data_shard(next(file_iter)), 0

    cuda_args = dict(device="cuda", non_blocking=True) if torch.cuda.is_available() else {}
    while True:
        if pos + batch_size + 1 >= len(tokens):
            tokens, pos = _load_data_shard(next(file_iter)), 0
        buf = tokens[pos + rank * local_batch_size:][:local_batch_size + 1]
        inputs = buf[:-1].to(dtype=torch.int32, **cuda_args).view(-1, 1024) # no sync on host side;
        targets = buf[1:].to(dtype=torch.int64, **cuda_args).view(-1, 1024) # H2D in another stream isn't helpful.
        pos += batch_size
        yield inputs, targets
