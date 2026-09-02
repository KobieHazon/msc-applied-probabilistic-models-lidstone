"""Helper for minimizing reading from the dataset files and parsing them"""

from collections import Counter
from pathlib import Path
from typing import Iterator


class ProcessedDocuments:
    """
    Helper class for reading and parsing the given dataset files
    """

    def __init__(self, documents_file_path: str | Path):
        """
        :param documents_file_path: file path for the dataset
        """
        self._documents_file_path = Path(documents_file_path)
        self._article_words_counter: Counter[str] | None = None
        self._article_words_count: int | None = None

    def iter_article_words(self) -> Iterator[str]:
        """
        Iterator for the words in the articles as presented in the dataset file
        :return: word in the article
        """
        with self._documents_file_path.open(encoding="utf-8") as documents_file:
            for line_num, line in enumerate(documents_file):
                if line_num % 4 == 2:
                    yield from line.split()

    def to_article_words_counter(self) -> Counter[str]:
        """
        :return: counter object for the words in the articles dataset
        """
        if self._article_words_counter is None:
            self._article_words_counter = Counter(self.iter_article_words())
        return self._article_words_counter

    def count_article_words(self) -> int:
        """
        :return: unique words in the articles dataset
        """
        if self._article_words_count is None:
            self._article_words_count = sum(self.to_article_words_counter().values())
        return self._article_words_count

    @property
    def file_name(self) -> str:
        return self._documents_file_path.name
