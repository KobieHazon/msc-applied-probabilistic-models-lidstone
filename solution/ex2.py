"""Command-line flow for training and evaluating Lidstone unigram models."""

import argparse
from collections import Counter
from itertools import islice
from pathlib import Path

from .development_set_preprocessing_stage import development_set_preprocessing
from .init_stage import init_stage
from .lidstone_model_training import lidstone_model_training
from .output_file_writer import OutputFileWriter
from .processed_documents import ProcessedDocuments
from .test_set_models_evaluation_stage import test_set_models_evaluation
from .unigram_language_model import UnigramLanguageModel
from .validate_models_stage import validate_models

# ratio to divide the development dataset to training and validation sets
TRAINING_VALIDATION_SET_RATIO = 0.9


def main(
    development_set_file_name: str | Path,
    test_set_file_name: str | Path,
    output_test_word: str,
    output_file_name: str | Path,
) -> None:
    """
    Main flow for the exercise, calls the different stages.
    :param development_set_file_name: file name for the development set
    :param test_set_file_name: file name for the test set
    :param output_test_word: the tested word whose metrics are outputted
    :param output_file_name: file name for the output file
    """
    development_processed_documents = ProcessedDocuments(development_set_file_name)
    test_processed_documents = ProcessedDocuments(test_set_file_name)

    training_set, validation_set = _get_training_validation_sets(development_processed_documents)
    training_set_counter = Counter(training_set)

    no_smoothing_model = UnigramLanguageModel.from_words_counter(training_set_counter)
    hundredth_smoothing_model = UnigramLanguageModel.from_words_counter(training_set_counter, 0.01)
    tenth_smoothing_model = UnigramLanguageModel.from_words_counter(training_set_counter, 0.10)
    one_smoothing_model = UnigramLanguageModel.from_words_counter(training_set_counter, 1.00)

    with OutputFileWriter(output_file_name) as output_writer:
        init_stage(
            output_writer,
            development_processed_documents,
            test_processed_documents,
            output_test_word,
        )
        development_set_preprocessing(output_writer, development_processed_documents)
        optimal_smooth_model = lidstone_model_training(
            output_writer,
            training_set,
            training_set_counter,
            validation_set,
            no_smoothing_model,
            hundredth_smoothing_model,
            tenth_smoothing_model,
            one_smoothing_model,
            output_test_word,
        )
        if not validate_models(
            {
                no_smoothing_model,
                hundredth_smoothing_model,
                tenth_smoothing_model,
                one_smoothing_model,
                optimal_smooth_model,
            }
        ):
            raise RuntimeError("one or more language models failed normalization")
        test_set_models_evaluation(output_writer, test_processed_documents, optimal_smooth_model)


def _get_training_validation_sets(
    development_processed_documents: ProcessedDocuments,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """
    Divides a file in the format of the exercise to training and validation sets
    :param development_processed_documents: handler for reading the processed articles file
    :return: tuple containing tuple for the training set and tuple for the validation set
    """
    total_articles_words = (
        development_processed_documents.count_article_words()
    )  # total words in the article
    training_set_size = round(
        total_articles_words * TRAINING_VALIDATION_SET_RATIO
    )  # use ratio and round
    article_words_iterator = (
        development_processed_documents.iter_article_words()
    )  # create iterator for creating sets

    training_set = tuple(
        islice(article_words_iterator, training_set_size)
    )  # use the iterator partly for the training
    validation_set = tuple(article_words_iterator)  # the rest is the validation set
    return training_set, validation_set


def cli() -> None:
    """Run the exercise from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("development_set", type=Path)
    parser.add_argument("test_set", type=Path)
    parser.add_argument("input_word")
    parser.add_argument("output_file", type=Path)
    args = parser.parse_args()

    for file_path in (args.development_set, args.test_set):
        if not file_path.is_file():
            parser.error(f"file does not exist: {file_path}")

    main(args.development_set, args.test_set, args.input_word, args.output_file)


if __name__ == "__main__":
    cli()
