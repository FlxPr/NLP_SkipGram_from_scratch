from __future__ import division
import argparse
import pandas as pd
from collections import Counter
from itertools import chain
import matplotlib.pyplot as plt
import json
import time

# useful stuff
import numpy as np
import random
from scipy.special import expit
from sklearn.preprocessing import normalize


path = 'data/training/news.en-00001-of-00100'  # First data file for coding and easy debugging
linux = False


def one_hot_encode(word_id, dimension):
    return np.array([[0] * word_id + [1] + [0] * (dimension - (word_id + 1))])


def sigmoid(x):
     return 1/(1 + np.exp(-x))


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
    def __init__(self, sentences, nEmbed=100, negativeRate=5, winSize=5, minCount=5, learning_rate=0.1,
                 input_weights=None, output_weights=None, w2id=None, vocab=None, id2frequency=None) :
        self.loss = None
        self.minCount = minCount  # Minimum word frequency to enter the dictionary
        self.winSize = winSize  # Window size for defining context
        self.negativeRate = negativeRate  # Number of negative samples to provide for each context word
        self.learning_rate = learning_rate
        self.trainWords = 0
        self.nEmbed = nEmbed
        self.id2frequency = id2frequency
        self.w2id = w2id
        self.vocab = vocab
        self.frequency_list = []
        self.sum_of_frequencies = 0
        self.id_list = 0
        self.input_weights = input_weights
        self.output_weights = output_weights
        self.id2frequency = id2frequency
        self.cached_samples = []

    def create_vocabulary(self):
        # Get the preprocessed sentences as a single list
        concatenated_sentences = chain.from_iterable(self.trainset)

        # Count the word occurrences
        word_counts = Counter(concatenated_sentences)

        # Filter down words by occurrence
        word_counts = {word: word_counts[word] for word in word_counts
                       if word_counts[word] >= self.minCount}

        return word_counts

    def create_word_mapping(self):  # Set index of word in unique words list as word id
        return {word: idx for idx, word in enumerate(list(self.vocab.keys()))}

    def create_negative_samples(self, size=100000):
        samples = list(np.random.choice(self.id_list,
                                        size=size,
                                        p=self.frequency_list))
        self.cached_samples.extend(samples)


    def sample(self, omit):
        global time_sampling
        global time_check_any_sampling

        tic = time.time()
        negative_samples = []
        while len(negative_samples) < self.negativeRate:
            if len(self.cached_samples) == 0:
                self.create_negative_samples()

            negative_sample = self.cached_samples.pop()
            if negative_sample not in omit:
                negative_samples.append(negative_sample)

        toc = time.time()
        time_sampling += toc - tic

        return negative_samples

    def train(self):
        global time_training
        tic = time.time()

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
        self.id2frequency = {self.w2id[word]: self.vocab[word] ** (3 / 4) for word in list(self.vocab.keys())}
        self.sum_of_frequencies = sum(list(self.id2frequency.values()))
        self.id_list = list(self.id2frequency.keys())
        self.frequency_list = list(map(lambda x: x / self.sum_of_frequencies, self.id2frequency.values()))

        print('Initializing weights')
        self.input_weights = np.random.randn(len(self.vocab), self.nEmbed) * np.sqrt(2 / len(self.vocab))
        self.output_weights = np.random.randn(self.nEmbed, len(self.vocab)) * np.sqrt(2 / self.nEmbed)

        for counter, sentence in enumerate(self.trainset):
            sentence = list(filter(lambda word: word in self.vocab, sentence))
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

                    negativeIds = self.sample(omit={wIdx, ctxtId})
                    self.trainWord(wIdx, ctxtId, negativeIds)
                    self.trainWords += 1

            if counter % 100 == 0:
                print(' > training %d of %d' % (counter, len(self.trainset)))
                print('elapsed time training {} seconds, sampling {} seconds'.format(int(time.time() - tic), time_sampling))
                # self.loss.append(self.accLoss / self.trainWords)
                # self.trainWords = 0
                # self.accLoss = 0.
        self.plot_embedding()

    def trainWord(self, wordId, contextId, negativeIds):
        word_embedding = self.input_weights[wordId, :]
        input_matrix_gradient = (sigmoid(word_embedding.dot(self.output_weights[:, contextId].reshape(self.nEmbed, 1))) - 1) * \
                                self.output_weights[:, contextId]

        output_matrix_gradient_positive_example = (sigmoid(word_embedding.dot(self.output_weights[:, contextId])) - 1) * word_embedding
        output_matrix_gradient_negative_examples = []
        for negative_id in negativeIds:
            output_matrix_gradient_negative_examples.append(sigmoid(word_embedding.dot(self.output_weights[:, negative_id])) * word_embedding)
            input_matrix_gradient += sigmoid(word_embedding.dot(self.output_weights[:, negative_id])) * \
                                     self.output_weights[:, negative_id]

        self.input_weights[wordId, :] -= self.learning_rate * input_matrix_gradient.flatten()
        self.output_weights[:, contextId] -= self.learning_rate * output_matrix_gradient_positive_example.flatten()
        for negative_id, gradient in zip(negativeIds, output_matrix_gradient_negative_examples):
            self.output_weights[:, negative_id] -= self.learning_rate * gradient.flatten()


    def save(self, path):
        parameters = {
            "input_weights": self.input_weights.tolist(),
            "output_weights": self.output_weights.tolist(),
            "vocab": self.vocab,
            "w2id": self.w2id,
            "id2frequency": self.id2frequency
        }

        print(str(parameters))
        json.dump(parameters, open(path, 'wt'))


    def similarity(self, word1, word2):
        """
            computes similiarity between the two words. unknown words are mapped to one common vector
        :param word1:
        :param word2:
        :return: a float \in [0,1] indicating the similarity (the higher the more similar)
        """

        if word1 in self.vocab and word2 in self.vocab:
            word1_embed = one_hot_encode(word1, len(self.vocab)).dot(self.input_weights)
            word2_embed = one_hot_encode(word2, len(self.vocab)).dot(self.input_weights)
            similarity = word1_embed.dot(word2_embed.T)/(np.linalg.norm(word1_embed) * np.linalg.norm(word2_embed)).item()
            return similarity.item()

        return 0  # TODO take care of unknown words (make average of words in corpus ?)


    @staticmethod
    def load(path):
        params = json.load(open(path))
        return SkipGram(sentences=[],
                        input_weights=params['input_weights'],
                        output_weights=params['output_weights'],
                        vocab=params['vocab'],
                        w2id=params['w2id'],
                        id2frequency=params['id2frequency'])

    def plot_embedding(self):
        if self.nEmbed != 2:
            return
        word_list = list(self.vocab.keys())
        print(word_list)

        word_embeddings = [one_hot_encode(self.w2id[word], len(self.vocab)).dot(self.input_weights) for word in word_list]
        fig, ax = plt.subplots()
        ax.scatter([word_embeddings[i][0][0] for i in range(len(word_list))], [word_embeddings[i][0][1] for i in range(len(word_list))])

        for i, letter in enumerate(word_list):
            ax.annotate(letter, (word_embeddings[i][0][0], word_embeddings[i][0][1]), size=20 if letter in ['felix', 'micha'] else 10)
        plt.title('Word embeddings')
        plt.show()


if __name__ == '__main__':
    if not linux:
        sentences = text2sentences(path)
        random.shuffle(sentences)
        sg_model = SkipGram(sentences, minCount=5, negativeRate=5, nEmbed=100, learning_rate=0.05)
        sg_model.train()
        sg_model.save('test_save_whole.json')
        load_sg = SkipGram.load('test_save_whole.json')
        print(load_sg.similarity('a', 'b'))


    elif linux:
        parser = argparse.ArgumentParser()
        parser.add_argument('--text', help='path containing training data', required=True)
        parser.add_argument('--model', help='path to store/read model (when training/testing)', required=True)
        parser.add_argument('--test', help='enters test mode', action='store_true')

        opts = parser.parse_args()

        if not opts.test:
            sentences = text2sentences(opts.text)
            random.shuffle(sentences)
            sg = SkipGram(sentences)
            sg.train()
            sg.save(opts.model)

        else:
            pairs = load_pairs(opts.text)

            sg = SkipGram.load(opts.model)
            for a, b, _ in pairs:
                # make sure this does not raise any exception, even if a or b are not in sg.vocab
                print(sg.similarity(a, b))

