"""Flow for the fourth stage - validating the models created in the third stage"""
from typing import Iterable

from unigram_language_model import UnigramLanguageModel


def validate_models(models: Iterable[UnigramLanguageModel]) -> bool:
    """
    Simply invoke the sanity_test function for each of the models
    :param models: to validate
    :return: True iff all models are valid
    """
    for model in models:
        if not model.sanity_test():
            return False
    return True
