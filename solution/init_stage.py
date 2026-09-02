"""Flow for the first stage (init stage) of the exercise"""
from consts import LANGUAGE_VOCABULARY_SIZE
from output_file_writer import OutputFileWriter
from processed_documents import ProcessedDocuments


def init_stage(output_writer: OutputFileWriter,
               development_processed_documents: ProcessedDocuments,
               test_processed_documents: ProcessedDocuments,
               output_test_word: str):
    """
    First stage of the exercise
    :param output_writer: writer to output file
    :param development_processed_documents: processed articles for the development dataset
    :param test_processed_documents: processed articles for the test dataset
    :param output_test_word: the word to be tested on the models later
    """
    # Output1: development set file name
    output_writer.write(development_processed_documents.file_name)
    # Output2: test set file name
    output_writer.write(test_processed_documents.file_name)
    # Output3: INPUT WORD
    output_writer.write(output_test_word)
    # Output4: output file name
    output_writer.write(output_writer.file_name)
    # Output5: language vocabulary size
    output_writer.write(LANGUAGE_VOCABULARY_SIZE)
    # Output6: P_uniform(Event = INPUT WORD), out of entire langauge vocabulary
    documents_words = development_processed_documents.to_article_words_counter()
    if output_test_word in documents_words:
        output_writer.write(1 / LANGUAGE_VOCABULARY_SIZE)
    else:
        output_writer.write(0)  # not found word has 0 uniform probability
