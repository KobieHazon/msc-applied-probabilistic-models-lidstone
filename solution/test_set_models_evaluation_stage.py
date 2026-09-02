"""Flow for the fifth and last stage of the exercise - evaluation on the test set"""

from .output_file_writer import OutputFileWriter
from .processed_documents import ProcessedDocuments
from .unigram_language_model import UnigramLanguageModel


def test_set_models_evaluation(
    output_writer: OutputFileWriter,
    test_processed_documents: ProcessedDocuments,
    optimal_language_model: UnigramLanguageModel,
) -> None:
    """
    Evaluates our optimal model on the test dataset
    :param output_writer: writer for the output values
    :param test_processed_documents: test dataset articles wrapper
    :param optimal_language_model: unigram language model with the optimal smoothing factor
    """
    # Output21: total number of events in the test set
    articles_words_count = (
        test_processed_documents.count_article_words()
    )  # number of words in the test dataset
    output_writer.write(articles_words_count)
    # Output22: The perplexity of the test set according to the Lidstone model with the λ that you chose
    test_set = tuple(
        test_processed_documents.iter_article_words()
    )  # all the words in the test dataset
    output_writer.write(optimal_language_model.calculate_perplexity(test_set))
