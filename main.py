import argparse
import logging
import json
import random
import numpy as np
import dynet as dy
from pathlib import Path

# create logger
logger = logging.getLogger("mylog")
logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG)


def loggerSeparator():
    logger.info("-----")


class Embedding:
    def __init__(self, fileName):
        loggerSeparator()
        logger.info("Reading word vectors " + fileName)
        self.words = self.readVectorsFromFile(fileName)
        logger.info("Creating UNK vector")
        self.UNK = self.createUNKVector()

        self.knownCounter = 0
        self.unknownCounter = 0

    def embedWord(self, word):
        if word in self.words:
            self.knownCounter += 1
            return self.words[word]

        self.unknownCounter += 1
        return self.UNK

    def embedWords(self, words):
        return np.array(map(self.embedWord, words))

    def createUNKVector(self):
        vectors = []
        for i, w in enumerate(self.words):
            if i > 100:
                break
            vectors.append(self.words[w])

        return np.average(vectors, axis=0)

    def readVectorsFromFile(self, fileName):
        words = {}
        with open(fileName, "r") as lines:
            for line in lines:
                vector = line.split()
                word = vector.pop(0)
                words[word] = np.array(map(float, vector))
        return words


LABELS = {
    u"entailment": 0,
    u"contradiction": 1,
    u"neutral": 2
}


class SNLIData:
    def __init__(self, type, fileName, embedding):
        loggerSeparator()
        logger.info("Reading data " + type + " " + fileName)

        self.data = []

        with open(fileName) as lines:
            for line in lines:
                datum = json.loads(line)
                gold = datum["gold_label"]
                if gold != u"-":
                    label = LABELS[gold]
                    sentence1 = embedding.embedWords(datum["sentence1"].lower().rstrip(".").split())
                    sentence2 = embedding.embedWords(datum["sentence2"].lower().rstrip(".").split())

                    self.data.append((sentence1, sentence2, label))

        logger.info(type + " size # sent " + str(len(self.data)))
        logger.info("Known words " + str(embedding.knownCounter) + " / Unknown words " + str(embedding.unknownCounter))


class Model:
    def __init__(self, embeddingSize, hiddenSize, labelsSize):
        self.model = dy.Model()
        self.trainer = dy.AdamTrainer(self.model)

        self.embeddingLinear = self.model.add_parameters((embeddingSize, hiddenSize))

        self.mlpF1 = self.model.add_parameters((hiddenSize, hiddenSize))
        self.mlpF2 = self.model.add_parameters((hiddenSize, hiddenSize))

        self.mlpG1 = self.model.add_parameters((2 * hiddenSize, hiddenSize))
        self.mlpG2 = self.model.add_parameters((hiddenSize, hiddenSize))

        self.mlpH1 = self.model.add_parameters((2 * hiddenSize, hiddenSize))
        self.mlpH2 = self.model.add_parameters((hiddenSize, hiddenSize))

        self.finaLinear = self.model.add_parameters((hiddenSize, labelsSize))

    def predict(self, data):
        return [self.forward(s1, s2) for (s1, s2, label) in data]

    def  accuracy(self, data):
        good = total = 0.0
        predicted = self.predict(data)
        golds = [label for (s1, s2, label) in data]

        for pred, gold in zip(predicted, golds):
            total += 1
            if pred == gold:
                good += 1

        return good / total

    def forward(self, sent1, sent2, label=None):
        """
        :param sent1: inputTensor
        :param sent2: inputTensor
        :param label: integer, range [0, 2]
        :return: loss
        """
        # Fix embedding
        eL = dy.parameter(self.embeddingLinear)
        sent1 = dy.inputTensor(sent1) * eL
        sent2 = dy.inputTensor(sent2) * eL

        # F step
        Lf1 = dy.parameter(self.mlpF1)
        Fsent1 = dy.rectify(dy.dropout(sent1, 0.2) * Lf1)
        Fsent2 = dy.rectify(dy.dropout(sent2, 0.2) * Lf1)
        Lf2 = dy.parameter(self.mlpF2)
        Fsent1 = dy.rectify(dy.dropout(Fsent1, 0.2) * Lf2)
        Fsent2 = dy.rectify(dy.dropout(Fsent2, 0.2) * Lf2)

        # Attention scoring
        score1 = Fsent1 * dy.transpose(Fsent2)
        prob1 = dy.softmax(score1)

        score2 = dy.transpose(score1)
        prob2 = dy.softmax(score2)

        # Align pairs using attention
        sent1Pairs = dy.concatenate_cols([sent1, prob1 * sent2])
        sent2Pairs = dy.concatenate_cols([sent2, prob2 * sent1])

        # G step
        Lg1 = dy.parameter(self.mlpG1)
        Gsent1 = dy.rectify(dy.dropout(sent1Pairs, 0.2) * Lg1)
        Gsent2 = dy.rectify(dy.dropout(sent2Pairs, 0.2) * Lg1)
        Lg2 = dy.parameter(self.mlpG2)
        Gsent1 = dy.rectify(dy.dropout(Gsent1, 0.2) * Lg2)
        Gsent2 = dy.rectify(dy.dropout(Gsent2, 0.2) * Lg2)

        # Sum
        Ssent1 = dy.sum_dim(Gsent1, [0])
        Ssent2 = dy.sum_dim(Gsent2, [0])

        concat = dy.transpose(dy.concatenate([Ssent1, Ssent2]))

        # H step
        Lh1 = dy.parameter(self.mlpH1)
        Hsent = dy.rectify(dy.dropout(concat, 0.2) * Lh1)
        Lh2 = dy.parameter(self.mlpH2)
        Hsent = dy.rectify(dy.dropout(Hsent, 0.2) * Lh2)

        # Final layer
        finalLayer = dy.parameter(self.finaLinear)
        # final = dy.softmax(dy.transpose(Hsent * finalLayer))
        final = dy.transpose(Hsent * finalLayer)

        if label != None:  # Label can be 0...
            return dy.pickneglogsoftmax(final, label)
        else:
            out = dy.softmax(final)
            chosen = np.argmax(out.npvalue())
            return chosen

    def save(self, modelFile):
        self.model.save(modelFile)

    def load(self, modelFile):
        self.model.populate(modelFile)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--train', help='training data file (jsonl)',
                        type=str, default='../snli_1.0/snli_1.0_train.jsonl')

    parser.add_argument('--dev', help='development data file (jsonl)',
                        type=str, default='../snli_1.0/snli_1.0_dev.jsonl')

    parser.add_argument('--test', help='test data file (jsonl)',
                        type=str, default='../snli_1.0/snli_1.0_test.jsonl')

    parser.add_argument('--w2v', help='pretrained word vectors file (word tab vector)',
                        type=str, default='deps.words')

    parser.add_argument('--embedding_size', help='word embedding size',
                        type=int, default=300)

    parser.add_argument('--epochs', help='training epochs',
                        type=int, default=25)

    parser.add_argument('--dev_interval', help='interval for development',
                        type=int, default=1)

    parser.add_argument('--display_interval', help='interval of display by batches',
                        type=int, default=5)

    parser.add_argument('--batch', help='size of batch',
                        type=int, default=20000)

    parser.add_argument('--model', help='path of model file (not include the name suffix',
                        type=str, default='model.save')

    parser.add_argument('--dynet-autobatch', help='dynet parameter',
                        type=int, default=1)

    parser.add_argument('--dynet-mem', help='dynet parameter',
                        type=int, default=8192)

    args = parser.parse_args()

    for arg in vars(args):
        logger.info(str(arg) + ' ' + str(getattr(args, arg)))

    embedding = Embedding(args.w2v)

    # load train/dev/test data
    trainData = SNLIData("train", args.train, embedding)
    devData = SNLIData("dev", args.dev, embedding)
    testData = SNLIData("test", args.test, embedding)

    model = Model(args.embedding_size, 300, len(LABELS))

    modelFileCache = Path(args.model)
    if modelFileCache.is_file():
        model.load(args.model)

    losses = []

    # accuracy = model.accuracy(testData.data)
    # logger.info("Test Accuracy: " + str(accuracy))

    # accuracy = model.accuracy(trainData.data[:10000])
    # logger.info("Train Accuracy: " + str(accuracy))

    loss = tagged = 0
    for EPOCH in range(args.epochs):
        loggerSeparator()
        logger.info("Starting epoch " + str(EPOCH))
        random.shuffle(trainData.data)

        errors = []
        dy.renew_cg()
        for i, (s1, s2, label) in enumerate(trainData.data, 1):
            if i % (args.batch * args.display_interval) == 0:
                avgLoss = loss / tagged
                losses.append(avgLoss)
                logger.info(str(EPOCH) + "/" + str(i) + ": " + str(avgLoss))
                loss = tagged = 0

                accuracy = model.accuracy(devData.data)
                logger.info("Dev Accuracy: " + str(accuracy))

                model.save(args.model)

            if i % args.batch == 0:
                errorsSum = dy.esum(errors)
                loss += errorsSum.value()
                tagged += args.batch

                errorsSum.backward()
                model.trainer.update()

                dy.renew_cg()
                errors = []

            errors.append(model.forward(s1, s2, label))
