
Benchmarking deep learning optimization with nanoGPT
====================================================
|Build Status| |Python 3.10+|

This benchmark is dedicated to evaluate new deep learning optimization methods
on the nanoGPT architecture.
The optimization problem is defined as in the original speedrun of nanoGPT (see [here](https://github.com/KellerJordan/modded-nanogpt/tree/master?tab=readme-ov-file)):
   - The training and validation is perfromed on [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb) -- Do not change the dataloaders.
   - The training is stopped once the validation loss is below ``3.28``.


$$\\min_{\\beta} f(X, \\beta),$$

where $X$ is the matrix of data and $\\beta$ is the optimization variable.

TODO:

- First we would like to reproduce the orignal result from the ``train_gpt2.py`` script.
- The goal is to make it easy to compare different optimization methods, not changes in the architecture.
- It should run this on 8xH00 GPus, but having the possibility to run
  this on single GPU would also be convenient.

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

.. |Build Status| image:: https://github.com/tomMoral/benchmark_nanogpt/actoiworkflows/main.yml/badge.svg
   :target: https://github.com/tomMoral/benchmark_nanogpt/actions
.. |Python 3.10+| image:: https://img.shields.io/badge/python-3.10%2B-blue
   :target: https://www.python.org/downloads/release/python-3100/
