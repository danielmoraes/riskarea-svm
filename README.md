Risk Area SVM
=============

Consider reading our [published paper](http://www.sciencedirect.com/science/article/pii/S1047320316300116) before playing with the code.

Getting Started
---------------

### Requirements

* g++
* make
* python 2.7x
* libsqlite3

### Installation

The only software that needs to be compiled is LIBSVM.

1. Go to `/path/to/rasvm/lib/libsvm-ext/` directory, and type `make`.

How to use
----------

1. Put your data inside `/path/to/rasvm/data/` directory.
2. Adjusts the global parameters inside the file `rasvm.py` to fit yours needs.
3. Go to `/path/to/rasvm/` directory, and run the Risk Area SVM with:
   * For a custom split: `python rasvm.py data/dataset/training.data data/dataset/testing.data`.
   * For a random 5x2cv: `python rasvm.py data/dataset/all.data`.
4. The results will be saved under `/path/to/rasvm/results/` directory.

Observations
------------

1. Your data needs to be in the LIBSVM format. You can use
   [this](http://www.csie.ntu.edu.tw/~cjlin/libsvm/faq.html#f307) to convert CSV
   data to LIBSVM format.
2. Your can save time by properly setting the global parameter `nr_local_worker`
   parameter inside the file `rasvm.py`. This sets the number of processes to be
   executed in parallel.
3. The positive class must be denoted by the label 1. The other labels will be
   automatically converted to -1 and considered as negative.
4. The data is automatically scaled to [-1, 1].
