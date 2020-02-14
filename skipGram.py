from __future__ import division
import argparse
import pandas as pd
from collections import Counter
from itertools import chain


# useful stuff
import numpy as np
from scipy.special import expit
from sklearn.preprocessing import normalize


path = 'data/training/news.en-00001-of-00100' # First data file for coding and easy debugging
linux = False


def text2sentences(path):
    # TODO: remove stop words, punctuation, etc.
    sentences = []
    with open(path, encoding="utf8") as f:
        for l in f:
            sentences.append(l.lower().split())
    return sentences


def loadPairs(path):
    data = pd.read_csv(path, delimiter='\t')
    pairs = zip(data['word1'], data['word2'], data['similarity'])
    return pairs


class SkipGram:
    def __init__(self, sentences, nEmbed=100, negativeRate=5, winSize=5, minCount=5):
        self.minCount = minCount,
        self.winSize = winSize,
        self.negativeRate = negativeRate,
        self.nEmbed = nEmbed,

        print('Initializing SkipGram model')
        self.trainset = sentences

        print('Creating vocabulary...')
        self.vocab = self.create_vocabulary()  # list of valid words
        print('Vocabulary of {} words created'.format(len(list(self.vocab))))

        print('Mapping words to ids')
        self.w2id = self.create_word_mapping()  # word to ID mapping

    def create_vocabulary(self):
        # Get the preprocessed sentences as a single list
        concatenated_list = chain.from_iterable(self.trainset)

        # Count the word occurrences
        word_counts = Counter(concatenated_list)

        # Filter down words by occurrence
        word_counts = {word: word_counts[word] for word in word_counts
                       if word_counts[word] >= self.minCount[0]}

        return list(word_counts.keys())

    def create_word_mapping(self):
        if len(list(self.vocab)) == 0:
            raise ValueError('The vocabulary is empty. Please initialize vocabulary by feeding a non-empty list of '
                             'sentences, or lower the minCount variable to take into account words with lower '
                             'frequency.')
        word_mapping = {}
        for idx, word in enumerate(list(self.vocab)):
            word_mapping[word] = idx
        return word_mapping

    def sample(self, omit):
        """samples negative words, ommitting those in set omit"""


        #raise NotImplementedError('this is easy, might want to do some preprocessing to speed up')


    def train(self):
        for counter, sentence in enumerate(self.trainset):
            sentence = filter(lambda word: word in self.vocab, sentence)
            for wpos, word in enumerate(sentence):
                # For every word and position in the sentence
                # Get the index of the word in the vocabulary
                wIdx = self.w2id[word]

                # Initializes a random window size for defining the word context (dynamic window size described in paper)
                winsize = np.random.randint(self.winSize) + 1
                # Computes the indexes of window start and stop to get the context words
                start = max(0, wpos - winsize)
                end = min(wpos + winsize + 1, len(sentence))

                for context_word in sentence[start:end]:
                    # Get the index of the context word in the vocabulary
                    ctxtId = self.w2id[context_word]
                    if ctxtId == wIdx:
                        continue  # skip if context word is the word itself
                    negativeIds = self.sample({wIdx, ctxtId})
                    self.trainWord(wIdx, ctxtId, negativeIds)
                    self.trainWords += 1

            if counter % 1000 == 0:
                print(' > training %d of %d' % (counter, len(self.trainset)))
                self.loss.append(self.accLoss / self.trainWords)
                self.trainWords = 0
                self.accLoss = 0.

    def trainWord(self, wordId, contextId, negativeIds):
        raise NotImplementedError('here is all the fun!')

    def save(self, path):
        raise NotImplementedError('implement it!')

    def similarity(self, word1, word2):
        """
            computes similiarity between the two words. unknown words are mapped to one common vector
        :param word1:
        :param word2:
        :return: a float \in [0,1] indicating the similarity (the higher the more similar)
        """
        raise NotImplementedError('implement it!')

    @staticmethod
    def load(path):
        raise NotImplementedError('implement it!')


if __name__ == '__main__':
    if not linux:
        print('Output of preprocessing :')
        txtToSentence_output = text2sentences(path)
        for i in range(5):
            print(txtToSentence_output[i])

        sg_model = SkipGram(txtToSentence_output)


    elif linux:
        parser = argparse.ArgumentParser()
        parser.add_argument('--text', help='path containing training data', required=True)
        parser.add_argument('--model', help='path to store/read model (when training/testing)', required=True)
        parser.add_argument('--test', help='enters test mode', action='store_true')

        opts = parser.parse_args()

        if not opts.test:
            sentences = text2sentences(opts.text)
            sg = SkipGram(sentences)
            sg.train()
            sg.save(opts.model)

        else:
            pairs = loadPairs(opts.text)

            sg = SkipGram.load(opts.model)
            for a, b, _ in pairs:
                # make sure this does not raise any exception, even if a or b are not in sg.vocab
                print(sg.similarity(a, b))



