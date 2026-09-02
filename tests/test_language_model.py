import math
from collections import Counter

import pytest

from solution.consts import LANGUAGE_VOCABULARY_SIZE
from solution.unigram_language_model import UnigramLanguageModel


def test_lidstone_probabilities_cover_the_language_vocabulary() -> None:
    model = UnigramLanguageModel.from_words_counter(Counter({"seen": 3, "word": 1}), 0.1)

    total_probability = (
        model.get_word_probability("seen")
        + model.get_word_probability("word")
        + (LANGUAGE_VOCABULARY_SIZE - 2) * model.get_word_probability("missing")
    )

    assert total_probability == pytest.approx(1.0)
    assert model.sanity_test()


def test_unsmoothed_model_has_infinite_perplexity_for_unseen_events() -> None:
    model = UnigramLanguageModel.from_words_counter(Counter({"seen": 2}))

    assert math.isinf(model.calculate_perplexity(("missing",)))


def test_negative_smoothing_is_rejected() -> None:
    with pytest.raises(ValueError, match="cannot be negative"):
        UnigramLanguageModel.from_words_counter(Counter({"seen": 1}), -0.1)
