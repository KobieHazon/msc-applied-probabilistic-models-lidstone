"""Helper for minimizing reading from the dataset files and parsing them"""
import os
from collections import Counter
from functools import lru_cache
from typing import Counter as CounterType, Iterator


class ProcessedDocuments:
    """
    Helper class for reading and parsing the given dataset files
    """

    def __init__(self, documents_file_path: str):
        """
        :param documents_file_path: file path for the dataset
        """
        self._documents_file = open(documents_file_path, "r")

    def iter_article_words(self) -> Iterator[str]:
        """
        Iterator for the words in the articles as presented in the dataset file
        :return: word in the article
        """
        self._documents_file.seek(0)  # start from the beginning of the file
        for line_num, line in enumerate(self._documents_file):
            if line_num % 4 == 2:  # only the article content line is interesting
                for word in line.split():
                    yield word

    @lru_cache(maxsize=1)
    def to_article_words_counter(self) -> CounterType[str]:
        """
        :return: counter object for the words in the articles dataset
        """
        return Counter(self.iter_article_words())

    @lru_cache(maxsize=1)
    def count_article_words(self) -> int:
        """
        :return: unique words in the articles dataset
        """
        return sum(self.to_article_words_counter().values())

    @property
    def file_name(self):
        return os.path.basename(self._documents_file.name)  # instructed to only output the basename
