# SNLI Decomposable Attention DyNet

This is a DyNet implementation fo decomposable attention for SNLI ported from the [PyTortch Implementation](https://github.com/libowen2121/SNLI-decomposable-attention),
of [this article](https://arxiv.org/abs/1606.01933).

## Usage

To use the code, run `main.py` with arguments of:
- w2v: An embedding file
- train: training file
- dev: dev file
- test: test file

You can find the train, dev, and test files in the [stanford database](https://nlp.stanford.edu/projects/snli/)

## Notes

- The code does not currently using actual batches.
- The code is still not working to 86%, but gets to 74.8%. (after like 8 epochs)
