from __future__ import division
import argparse
import pandas as pd
from collections import Counter
from itertools import chain


# useful stuff
import numpy as np
import random
from scipy.special import expit
from sklearn.preprocessing import normalize


path = 'data/training/news.en-00001-of-00100'  # First data file for coding and easy debugging
linux = False


def text2sentences(path):
    # TODO: remove stop words ?
    # No need to take care of rare weird words like bqb4645 which will not be in dictionary due to minimum word count
    sentences = []
    with open(path, encoding="utf8") as file:
        for sentence in file:
            preprocessed_sentence = sentence.lower().split()

            # Gets rid of small words and punctuation (length<2) and numbers like dates that may show up frequently
            preprocessed_sentence = list(filter(lambda x: not x.isdigit() and len(x) > 2, preprocessed_sentence))

            sentences.append(preprocessed_sentence)
    return sentences


def load_pairs(path):
    data = pd.read_csv(path, delimiter='\t')
    pairs = zip(data['word1'], data['word2'], data['similarity'])
    return pairs


class SkipGram:
    def __init__(self, sentences, nEmbed=100, negativeRate=5, winSize=5, minCount=5):
        self.loss = None
        self.minCount = minCount,  # Minimum word frequency to enter the dictionary
        self.winSize = winSize,  # Window size for defining context
        self.negativeRate = negativeRate,  # Number of negative samples to provide for each context word (I assume)
        self.nEmbed = nEmbed,  # idk what that is that thing that it is what is this

        print('Initializing SkipGram model')
        self.trainset = sentences

        print('Creating vocabulary...')
        self.vocab = self.create_vocabulary()  # list of valid words
        if len(list(self.vocab)) == 0:
            raise ValueError('The vocabulary is empty. Please initialize vocabulary by feeding a non-empty list of '
                             'sentences, or lower the minCount variable to take into account words with lower '
                             'frequency.')

        print('Mapping words to ids')
        self.w2id = self.create_word_mapping()  # word to ID mapping

        # Map word ID to word frequency (speeds up computation for negative sampling)
        print('Mapping ids to frequencies')
        self.id2frequency = {self.w2id[word]: self.vocab[word] ** (3/4) for word in list(self.vocab.keys())}
        self.sum_of_frequencies = sum(list(self.id2frequency.values()))

    def create_vocabulary(self):
        # Get the preprocessed sentences as a single list
        concatenated_list = chain.from_iterable(self.trainset)

        # Count the word occurrences
        word_counts = Counter(concatenated_list)

        # Filter down words by occurrence
        word_counts = {word: word_counts[word] for word in word_counts
                       if word_counts[word] >= self.minCount[0]}

        return word_counts

    def create_word_mapping(self):
        word_mapping = {}
        for idx, word in enumerate(list(self.vocab.keys())):
            word_mapping[word] = idx
        return word_mapping

    def sample(self, omit):
        # Sample words from dictionary based on weighted probability, inspired from
        # https://stackoverflow.com/questions/40927221/how-to-choose-keys-from-a-python-dictionary-based-on-weighted-probability

        sum_of_frequencies = self.sum_of_frequencies
        for omit_id in omit:
            try:
                sum_of_frequencies -= self.id2frequency[omit_id]
            except KeyError:
                pass  # Omit word not present in vocabulary, no need to take it into account for negative sampling

        sample_candidates = {id: freq for id, freq in self.id2frequency.items() if id not in omit}
        negative_ids = []

        for sample_number in range(self.negativeRate[0]):
            random_word_index = random.random() * sum_of_frequencies
            total = 0
            for id, probability in sample_candidates.items():
                total += probability
                if random_word_index <= total:
                    negative_ids.append(id)
                    break

        return negative_ids

    def train(self):
        for counter, sentence in enumerate(self.trainset):
            sentence = filter(lambda word: word in self.vocab, sentence)
            for wpos, word in enumerate(sentence):
                # For every word and position in the sentence
                # Get the index of the word in the vocabulary
                wIdx = self.w2id[word]

                # Initializes a random window size for defining the word context
                # (dynamic window size described in paper)
                winsize = np.random.randint(self.winSize) + 1

                # Computes the indexes of window start and stop to get the context words
                start = max(0, wpos - winsize)
                end = min(wpos + winsize + 1, len(sentence))

                for context_word in sentence[start:end]:
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

        sentence = ' '.join(['haha ' * 100 + 'hihi ' * 20 + 'hoho ' * 5 + '.']).split()
        random.shuffle(sentence)

        sg_model = SkipGram([sentence], minCount=1, negativeRate=15)

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
            pairs = load_pairs(opts.text)

            sg = SkipGram.load(opts.model)
            for a, b, _ in pairs:
                # make sure this does not raise any exception, even if a or b are not in sg.vocab
                print(sg.similarity(a, b))

