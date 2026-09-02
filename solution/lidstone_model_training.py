"""Train and select a Lidstone-smoothed unigram language model."""

from collections import Counter
from collections.abc import Collection

from .consts import (
    SMOOTHING_FACTOR_MAX,
    SMOOTHING_FACTOR_MIN,
    SMOOTHING_FACTOR_STEP,
    UNSEEN_WORD,
)
from .output_file_writer import OutputFileWriter
from .unigram_language_model import UnigramLanguageModel


def lidstone_model_training(
    output_writer: OutputFileWriter,
    training_set: Collection[str],
    training_set_counter: Counter[str],
    validation_set: Collection[str],
    no_smoothing_model: UnigramLanguageModel,
    hundredth_smoothing_model: UnigramLanguageModel,
    tenth_smoothing_model: UnigramLanguageModel,
    one_smoothing_model: UnigramLanguageModel,
    output_test_word: str,
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
        _find_optimal_smoothing_factor(training_set_counter, validation_set),
        2,
    )
    output_writer.write(optimal_smoothing_factor)
    # Output20: The minimized perplexity on the validation set using the best value you found for λ
    optimal_model = UnigramLanguageModel.from_words_counter(
        training_set_counter, optimal_smoothing_factor
    )
    output_writer.write(optimal_model.calculate_perplexity(validation_set))
    return optimal_model


def _find_optimal_smoothing_factor(
    training_set_counter: Counter[str],
    validation_set: Collection[str],
) -> float:
    """
    Find the two-decimal smoothing factor with the lowest validation perplexity.
    :param training_set_counter: counter for the training set
    :param validation_set: validation set for calculating the perplexity on
    :return: optimal smoothing factor for the validation set perplexity
    """
    first_step = round(SMOOTHING_FACTOR_MIN / SMOOTHING_FACTOR_STEP)
    last_step = round(SMOOTHING_FACTOR_MAX / SMOOTHING_FACTOR_STEP)
    candidates = (step * SMOOTHING_FACTOR_STEP for step in range(first_step, last_step + 1))
    return min(
        candidates,
        key=lambda smoothing_factor: UnigramLanguageModel.from_words_counter(
            training_set_counter, smoothing_factor
        ).calculate_perplexity(validation_set),
    )
