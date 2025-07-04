from benchopt import BaseDataset, safe_import_context
from benchopt.config import get_data_path


# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:

    from benchmark_utils.model_gpt2 import GPT, GPTConfig


def download_data(data_dir, n_chunks=104):

    from huggingface_hub import hf_hub_download

    # Download the GPT-2 tokens of Fineweb10B from huggingface. This
    # saves about an hour of startup time compared to regenerating them.
    for i in range(n_chunks):
        chunk = "val" if i == 0 else "train"
        fname = f"fineweb_{chunk}_{i:06d}.bin"
        if not (data_dir / fname).exists():
            hf_hub_download(repo_id="kjj0/fineweb10B-gpt2", filename=fname,
                            repo_type="dataset", local_dir=data_dir)


# All datasets must be named `Dataset` and inherit from `BaseDataset`
class Dataset(BaseDataset):

    # Name to select the dataset in the CLI and to display the results.
    name = "Fine-web"

    # List of parameters to generate the datasets. The benchmark will consider
    # the cross product for each key in the dictionary.
    # Any parameters 'param' defined here is available as `self.param`.
    parameters = {
        'n_chunks': [2],
    }

    # List of packages needed to run the dataset. See the corresponding
    # section in objective.py
    requirements = []

    def get_data(self):

        data_dir = get_data_path("fineweb10B")
        download_data(data_dir, n_chunks=self.n_chunks)

        # from scratch (random weights)
        config = GPTConfig(
            vocab_size=50257, n_layer=12, n_head=6, n_embd=768,
            # max_seq_len=4*64*1024 - This is for Rotary Positional Embedding
        )
        model = GPT(config)

        # The dictionary defines the keyword arguments for `Objective.set_data`
        return dict(
            train_files=str(data_dir / "fineweb_train_*.bin"),
            val_files=str(data_dir / "fineweb_val_*.bin"),
            model=model,
        )
