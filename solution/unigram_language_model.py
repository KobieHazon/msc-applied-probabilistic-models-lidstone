"""Representation of the unigram language model"""

import math
from collections import Counter
from collections.abc import Collection, Mapping

from .consts import LANGUAGE_VOCABULARY_SIZE, ROUND_DIGITS_CNT


class UnigramLanguageModel:
    """
    Class representing the unigram language model
    """

    def __init__(
        self,
        words_probabilities: Mapping[str, float],
        unseen_word_probability: float,
    ):
        """
        :param words_probabilities: probability of every word trained on
        :param unseen_word_probability: probability of a word not trained on
        """
        self._words_probabilities = words_probabilities
        self._unseen_word_probability = unseen_word_probability

    def get_word_probability(self, word: str) -> float:
        """
        :return: probability of word
        """
        if word in self._words_probabilities:
            return self._words_probabilities[word]
        return self._unseen_word_probability

    def calculate_perplexity(self, on_set: Collection[str]) -> float:
        """
        Calculate the perplexity of this model on a given set
        :param on_set: to calculate perplexity on
        :return: perplexity of model on set
        """
        # logarithmic sum of probabilities
        if not on_set:
            raise ValueError("perplexity requires at least one event")
        probabilities = (self.get_word_probability(word) for word in on_set)
        try:
            logarithmic_sum = sum(math.log2(probability) for probability in probabilities)
        except ValueError:
            return math.inf
        return math.pow(2, (-1 / len(on_set)) * logarithmic_sum)  # as power of 2

    @classmethod
    def from_words_counter(
        cls, words_counter: Counter[str], smoothing_factor: float = 0
    ) -> "UnigramLanguageModel":
        """
        Create an unigram language model from a word counter instance and a smoothing factor
        :param words_counter: counter to use for word frequencies
        :param smoothing_factor: smoothing factor for lidstone smoothing
        :return: unigram language model fitting the arguments
        """
        word_probabilities = {}
        word_space_size = sum(words_counter.values())

        for word, word_cnt in (+words_counter).items():  # only the words with positive frequency
            word_probabilities[word] = cls.calculate_mle_probability(
                word_cnt, word_space_size, smoothing_factor
            )
        # probability of unseen word
        unseen_word_probability = cls.calculate_mle_probability(
            0, word_space_size, smoothing_factor
        )

        return cls(word_probabilities, unseen_word_probability)

    @staticmethod
    def calculate_mle_probability(
        word_cnt: int, word_space_size: int, smoothing_factor: float
    ) -> float:
        """
        Calculate the MLE probability for a word
        :param word_cnt: number of times the word appeared
        :param word_space_size: amount of words in the training set
        :param smoothing_factor: factor to do lidstone smoothing by
        :return: MLE probability for word with said metrics
        """
        if smoothing_factor < 0:
            raise ValueError("smoothing factor cannot be negative")
        denominator = word_space_size + smoothing_factor * LANGUAGE_VOCABULARY_SIZE
        if denominator == 0:
            raise ValueError("cannot train a model from an empty event collection")
        return (word_cnt + smoothing_factor) / denominator

    @property
    def word_space_size(self) -> int:
        return len(self._words_probabilities)

    def sanity_test(self) -> bool:
        """
        Tests the probabilities of the model sum up to 1
        :return: True iff sum of probabilities is 1
        """
        unseen_words_count = LANGUAGE_VOCABULARY_SIZE - len(self._words_probabilities)
        probabilities_sum = unseen_words_count * self._unseen_word_probability + sum(
            self._words_probabilities.values()
        )
        return round(probabilities_sum, ROUND_DIGITS_CNT) == 1
