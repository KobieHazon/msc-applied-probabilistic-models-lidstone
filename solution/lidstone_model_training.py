"""Flow for the third and largest stage of the exercise - lidstone model training."""
import random
from typing import Collection, Counter, Tuple

from consts import SMOOTHING_FACTOR_GRADIENT_DECENT_LEARNING_RATE, SMOOTHING_FACTOR_GRADIENT_THRESHOLD, UNSEEN_WORD
from output_file_writer import OutputFileWriter
from unigram_language_model import UnigramLanguageModel


def lidstone_model_training(
        output_writer: OutputFileWriter,
        training_set: Collection[str],
        training_set_counter: Counter[str],
        validation_set: Collection[str],
        no_smoothing_model: UnigramLanguageModel,
        hundredth_smoothing_model: UnigramLanguageModel,
        tenth_smoothing_model: UnigramLanguageModel,
        one_smoothing_model: UnigramLanguageModel,
        output_test_word: str
) -> UnigramLanguageModel:
    """
    Flow for the third and largest stage of the exercise - lidstone model training
    :param output_writer: writer to the output file
    :param training_set: set of words for training of models
    :param training_set_counter: counter for the training set, useful for the training
    :param validation_set: set of words for validation of the models
    :param no_smoothing_model: lidstone model without smoothing
    :param hundredth_smoothing_model: lidstone model with smoothing of 0.01
    :param tenth_smoothing_model: lidstone model with smoothing of 0.1
    :param one_smoothing_model: lidstone model with smoothing of 1.0
    :param output_test_word: the word to test the models on
    :return: the lidstone model with the optimal smoothing factor
    """
    # Output8: number of events in the validation set
    output_writer.write(len(validation_set))
    # Output9: number of events in the training set
    output_writer.write(len(training_set))
    # Output 10: the number of different events in the training set
    output_writer.write(no_smoothing_model.word_space_size)
    # Output11: the number of times the event INPUT WORD appears in the training set
    output_writer.write(training_set_counter[output_test_word])
    # Output12: P(Event = INPUT WORD) the Maximum Likelihood Estimate (MLE) based on the training set
    output_writer.write(no_smoothing_model.get_word_probability(output_test_word))
    # Output13: P(Event = ’unseen-word’) the Maximum Likelihood Estimate (MLE) based on the training set
    output_writer.write(no_smoothing_model.get_word_probability(UNSEEN_WORD))
    # Output14: P(Event = INPUT WORD) as estimated by your model using λ = 0.10
    output_writer.write(tenth_smoothing_model.get_word_probability(output_test_word))
    # Output15: P(Event = ’unseen-word’) as estimated by your model using λ = 0.10
    output_writer.write(tenth_smoothing_model.get_word_probability(UNSEEN_WORD))
    # Output16: The perplexity on the validation set using λ = 0.01
    output_writer.write(hundredth_smoothing_model.calculate_perplexity(validation_set))
    # Output17: The perplexity on the validation set using λ = 0.10
    output_writer.write(tenth_smoothing_model.calculate_perplexity(validation_set))
    # Output18: The perplexity on the validation set using λ = 1.00
    output_writer.write(one_smoothing_model.calculate_perplexity(validation_set))
    # Output 19: The value of λ that you found to minimize the perplexity on the validation set
    optimal_smoothing_factor = round(
        _find_optimal_smoothing_factor(training_set_counter, validation_set, (0.0001, 2)),  # range instructed
        2  # instructed for 2 digits after the point
    )
    output_writer.write(optimal_smoothing_factor)
    # Output20: The minimized perplexity on the validation set using the best value you found for λ
    optimal_model = UnigramLanguageModel.from_words_counter(training_set_counter, optimal_smoothing_factor)
    output_writer.write(optimal_model.calculate_perplexity(validation_set))
    return optimal_model


def _find_optimal_smoothing_factor(
        training_set_counter: Counter[str],
        validation_set: Collection[str],
        in_range: Tuple[float, float]
) -> float:
    """
    Performs gradient descent on the smoothing factor to find the one with the minimum perplexity
    :param training_set_counter: counter for the training set
    :param validation_set: validation set for calculating the perplexity on
    :param in_range: range to perform gradient descent in
    :return: optimal smoothing factor for the validation set perplexity
    """

    def get_gradient(point_x: float, step: float = 0.00001) -> float:
        """
        Calculates the gradient in a point based on a very close point to it.
        :param point_x: point of the calculated gradient
        :param step: closest point as helper for gradient calculation
        :return: gradient of perplexity graph at point point_x
        """
        point_model = UnigramLanguageModel.from_words_counter(training_set_counter, point_x)
        point_y = point_model.calculate_perplexity(validation_set)
        other_point_x = point_x - step
        other_point_model = UnigramLanguageModel.from_words_counter(training_set_counter, other_point_x)
        other_point_y = other_point_model.calculate_perplexity(validation_set)
        return (point_y - other_point_y) / (point_x - other_point_x)

    curr_point = random.uniform(*in_range)  # start from random point in the range
    # while the gradient is larger than the threshold
    while abs(gradient := get_gradient(curr_point)) > SMOOTHING_FACTOR_GRADIENT_THRESHOLD:
        # move the point in the opposite side of the gradient
        curr_point = curr_point - SMOOTHING_FACTOR_GRADIENT_DECENT_LEARNING_RATE * gradient

    return curr_point
