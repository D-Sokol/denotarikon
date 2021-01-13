from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import PreTrainedModel, PreTrainedTokenizer
from typing import Tuple

from exceptions import IncorrectInput


def get_model(path: str = None) -> Tuple[PreTrainedTokenizer, PreTrainedModel]:
    try:
        if path is not None:
            tokenizer = GPT2Tokenizer.from_pretrained(path, add_prefix_space=True)
            model = GPT2LMHeadModel.from_pretrained(path).train(False)
            return tokenizer, model
    except OSError as err:
        pass

    # Path is not provided or loading failed.
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2', add_prefix_space=True)
    model = GPT2LMHeadModel.from_pretrained('gpt2').train(False)

    if path is not None:
        tokenizer.save_pretrained(path)
        model.save_pretrained(path)

    return tokenizer, model

