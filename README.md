# SNLI Decomposable Attention DyNet

This is a DyNet implementation fo decomposable attention for SNLI partially based on the [PyTortch Implementation](https://github.com/libowen2121/SNLI-decomposable-attention),
of [this article](https://arxiv.org/abs/1606.01933).

## Usage

To use the code, run `main.py` with arguments of:
- w2v: An embedding file
- train: training file
- dev: dev file
- test: test file

So for example:
```bash
python main.py --w2v=deps.words --train=../snli_1.0/snli_1.0_train.jsonl --dev=../snli_1.0/snli_1.0_traindev.jsonl --test=../snli_1.0/snli_1.0_test.jsonl
```

You can find the train, dev, and test files in the [stanford database](https://nlp.stanford.edu/projects/snli/)
