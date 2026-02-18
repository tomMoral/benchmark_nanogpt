
Benchmarking deep learning optimization with nanoGPT
====================================================
|Build Status| |Python 3.10+|

This benchmark is dedicated to evaluate new deep learning optimization methods
on the nanoGPT architecture.
The optimization problem is defined as in the original speedrun of nanoGPT (see `modded nanogpt`_):

- The training and validation is perfromed on `FineWeb`_ -- Do not change the dataloaders.
- The training is stopped once the validation loss is below ``3.28``. (Still todo)


For now, the repository contains a single solver, Adam, and run on CPU.
The dataloaders are working but with fixed sequence length of 128 tokens.
We used the original code from nanoGPT (`GPT2 from llm.c`_), but use the simple dataloader from `modded-nanogpt`_.

TODO:

- Tweak the dataloaders to make it more efficient/less error prone.
- See if we want to add imporevments to the architecture (QK-norm, Rotary embeddings, etc.).

Install
--------

This benchmark can be run using the following commands:

.. code-block::

   $ pip install -U benchopt
   $ git clone https://github.com/tomMoral/benchmark_nanogpt
   $ benchopt run benchmark_nanogpt

Apart from the problem, options can be passed to ``benchopt run``, to restrict the benchmarks to some solvers or datasets, e.g.:

.. code-block::

	$ benchopt run benchmark_nanogpt -s solver1 -d dataset2 --max-runs 10 --n-repetitions 10


Use ``benchopt run -h`` for more details about these options, or visit https://benchopt.github.io/api.html.

.. |Build Status| image:: https://github.com/tomMoral/benchmark_nanogpt/actions/workflows/main.yml/badge.svg
   :target: https://github.com/tomMoral/benchmark_nanogpt/actions
.. |Python 3.10+| image:: https://img.shields.io/badge/python-3.10%2B-blue
   :target: https://www.python.org/downloads/release/python-3100/

.. _FineWeb: https://huggingface.co/datasets/HuggingFaceFW/fineweb
.. _modded nanogpt: https://github.com/KellerJordan/modded-nanogpt
.. _GPT2 from llm.c: https://github.com/karpathy/llm.c
