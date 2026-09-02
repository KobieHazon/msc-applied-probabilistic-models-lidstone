"""Main flow of the exercise, calls the different stages and does validation for the invocation"""
import os
from collections import Counter
from itertools import islice
from sys import argv
from typing import Tuple

from development_set_preprocessing_stage import development_set_preprocessing
from init_stage import init_stage
from lidstone_model_training import lidstone_model_training
from output_file_writer import OutputFileWriter
from processed_documents import ProcessedDocuments
from test_set_models_evaluation_stage import test_set_models_evaluation
from unigram_language_model import UnigramLanguageModel
from validate_models_stage import validate_models

# message shown when incorrect program input is received
PROGRAM_ARGS_HELP_MSG = ("Input format must be: \n"
                         "python ex2.py "
                         "<development_processed_documents> <test_processed_documents> <INPUT_WORD> <output_file_name>")

# ratio to divide the development dataset to training and validation sets
TRAINING_VALIDATION_SET_RATIO = 0.9


def main(development_set_file_name: str, test_set_file_name: str, output_test_word: str, output_file_name: str):
    """
    Main flow for the exercise, calls the different stages.
    :param development_set_file_name: file name for the development set
    :param test_set_file_name: file name for the test set
    :param output_test_word: the tested word whose metrics are outputted
    :param output_file_name: file name for the output file
    """
    output_writer = OutputFileWriter(output_file_name)  # writer to output file
    development_processed_documents = ProcessedDocuments(development_set_file_name)  # accessing the development dataset
    test_processed_documents = ProcessedDocuments(test_set_file_name)  # accessing the test dataset

    training_set, validation_set = _get_training_validation_sets(development_processed_documents)  # divide development
    training_set_counter = Counter(training_set)  # create counter for the training set, useful for training

    # create models with different smoothing factors
    no_smoothing_model = UnigramLanguageModel.from_words_counter(training_set_counter)
    hundredth_smoothing_model = UnigramLanguageModel.from_words_counter(training_set_counter, 0.01)
    tenth_smoothing_model = UnigramLanguageModel.from_words_counter(training_set_counter, 0.10)
    one_smoothing_model = UnigramLanguageModel.from_words_counter(training_set_counter, 1.00)

    # invoke the different stages with the arguments they need
    init_stage(output_writer, development_processed_documents, test_processed_documents, output_test_word)  # stage 1
    development_set_preprocessing(output_writer, development_processed_documents)  # stage 2
    optimal_smooth_model = lidstone_model_training(output_writer,
                                                   training_set,
                                                   training_set_counter,
                                                   validation_set,
                                                   no_smoothing_model,
                                                   hundredth_smoothing_model,
                                                   tenth_smoothing_model,
                                                   one_smoothing_model,
                                                   output_test_word)  # stage 3
    assert validate_models({no_smoothing_model,
                            hundredth_smoothing_model,
                            tenth_smoothing_model,
                            one_smoothing_model,
                            optimal_smooth_model})  # stage 4
    test_set_models_evaluation(output_writer, test_processed_documents, optimal_smooth_model)  # stage 5


def _get_training_validation_sets(
        development_processed_documents: ProcessedDocuments,
) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    """
    Divides a file in the format of the exercise to training and validation sets
    :param development_processed_documents: handler for reading the processed articles file
    :return: tuple containing tuple for the training set and tuple for the validation set
    """
    total_articles_words = development_processed_documents.count_article_words()  # total words in the article
    training_set_size = round(total_articles_words * TRAINING_VALIDATION_SET_RATIO)  # use ratio and round
    article_words_iterator = development_processed_documents.iter_article_words()  # create iterator for creating sets

    training_set = tuple(islice(article_words_iterator, training_set_size))  # use the iterator partly for the training
    validation_set = tuple(article_words_iterator)  # the rest is the validation set
    return training_set, validation_set


if __name__ == "__main__":
    if len(argv) != 5:  # make sure 5 arguments were received
        raise ValueError(PROGRAM_ARGS_HELP_MSG)
    _, development_set_filename, test_set_filename, input_word, output_filename = argv

    for file_name in (development_set_filename, test_set_filename):  # make sure the files needed exist
        if not os.path.isfile(file_name):
            raise ValueError(f"file {file_name} does not exist")
    main(development_set_filename, test_set_filename, input_word, output_filename)
