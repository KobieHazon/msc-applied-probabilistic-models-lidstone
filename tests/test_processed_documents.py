from solution.processed_documents import ProcessedDocuments


def test_only_article_body_words_are_read(tmp_path) -> None:
    dataset = tmp_path / "documents.txt"
    dataset.write_text(
        "<TRAIN 1 topic>\n\nfirst article words\n\n<TRAIN 2 topic>\n\nsecond article\n\n",
        encoding="utf-8",
    )

    documents = ProcessedDocuments(dataset)

    assert tuple(documents.iter_article_words()) == (
        "first",
        "article",
        "words",
        "second",
        "article",
    )
    assert documents.count_article_words() == 5
    assert documents.to_article_words_counter()["article"] == 2
    assert documents.file_name == "documents.txt"
