"""Implementation of the second stage of the exercise - development set preprocessing."""

from .output_file_writer import OutputFileWriter
from .processed_documents import ProcessedDocuments


def development_set_preprocessing(
    output_writer: OutputFileWriter,
    development_processed_documents: ProcessedDocuments,
) -> None:
    """
    Implements the development set preprocessing stage.
    :param output_writer: writes to the output file
    :param development_processed_documents: object for accessing the development set
    """
    # Output7: total number of events in the development set |S|
    articles_words_count = development_processed_documents.count_article_words()
    output_writer.write(articles_words_count)
