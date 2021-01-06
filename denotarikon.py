#!/usr/bin/env python3

import torch
import numpy as np
import string
import sys
from transformers import GPT2Tokenizer, GPT2LMHeadModel

from parser import get_args


class IncorrectInput(ValueError):
    pass


def letter_index(letter, alphabet=string.ascii_lowercase):
    return alphabet.index(letter)


def is_starting(token, alphabet=string.ascii_lowercase):
    """
    Determines if the token starts a new word
    """
    return len(token) > 1 and token.startswith(' ') and token[1].lower() in alphabet


def is_continuing(token, alphabet=string.ascii_lowercase):
    """
    Determines if the token continue previous word
    """
    return token and not token[0].isspace() and (token[0] in alphabet or token[0].lower() not in alphabet)


def get_masks(tokenizer, alphabet=string.ascii_lowercase):
    all_tokens = [tokenizer.decode(i) for i in range(tokenizer.vocab_size)]
    mask_allowed = torch.zeros(tokenizer.vocab_size, len(alphabet), dtype=bool)
    mask_starting = torch.zeros(tokenizer.vocab_size, dtype=bool)
    for token, mask_row_a, mask_row_s in zip(all_tokens, mask_allowed, mask_starting):
        if is_starting(token):
            mask_row_a[letter_index(token[1].lower())] = True
            # Plain assignment or `mask_row_s[:] = True` are not applicable to scalars,
            #  thus I use operator |=, which works just like inplace assignment.
            mask_row_s |= True
        elif is_continuing(token):
            mask_row_a[:] = True
    return mask_allowed, mask_starting


def target_letters_covered(target, start_tokens, tokenizer):
    if not target:
        raise IncorrectInput("Target string cannot be empty")

    chars_allowed = set(string.ascii_letters)
    chars_given = set(target)
    if not chars_given.issubset(chars_allowed):
        # The second term produces something like " '1','&',' ' ".
        raise IncorrectInput("Target string cannot contain following symbols: " + ",".join(map(repr, chars_given - chars_allowed)))

    target_letter_generated = 0
    for ix in start_tokens:
        token = tokenizer.decode(ix)
        if not is_starting(token):
            continue

        if target_letter_generated == len(target):
            raise IncorrectInput("Initial phrase is too long to match the target")
        if target[target_letter_generated].lower() != token[1].lower():
            raise IncorrectInput("Initial phrase does not match the target")
        target_letter_generated += 1
    return target_letter_generated


def generate(target, start_tokens, model, tokenizer,
             device='cpu', p_threshold=0.95, max_nostarting_token=5, temperature=1.0):
    target = target.lower()
    tokens = start_tokens.copy()
    with torch.no_grad():
        mask_allowed, mask_starting = get_masks(tokenizer)
        target_letter_generated = target_letters_covered(target, start_tokens, tokenizer)
        if start_tokens:
            result = model(torch.tensor(tokens, device=device)[None], past_key_values=None)
            next_logits, past = result['logits'][0, -1, :], result['past_key_values']
            rest_nostarting_tokens = max_nostarting_token
        else:
            next_logits, past = torch.zeros(tokenizer.vocab_size), None
            rest_nostarting_tokens = 0

        while target_letter_generated < len(target):
            next_logits[~mask_allowed[:, letter_index(target[target_letter_generated])]] = -np.inf
            if not rest_nostarting_tokens:
                next_logits[~mask_starting] = -np.inf
            next_probas = torch.softmax(next_logits / temperature, dim=-1).cpu()

            sorted_p, sorted_ix = torch.sort(next_probas, descending=True)
            cumulative_p = torch.cumsum(sorted_p, dim=-1)

            # Number of possible choices for next token, calculated as minimal n
            #  such that sum of probabilities of the first n tokens exceeds p_threshold
            n_tokens_next = np.argmax(cumulative_p.numpy() > p_threshold) + 1

            sorted_p = sorted_p[:n_tokens_next]
            sorted_p /= cumulative_p[n_tokens_next-1]
            ix_ix = np.random.choice(n_tokens_next, p=sorted_p.numpy())
            next_ix = sorted_ix[ix_ix]
            tokens.append(next_ix.item())
            if mask_starting[next_ix]:
                target_letter_generated += 1
                rest_nostarting_tokens = max_nostarting_token
            else:
                rest_nostarting_tokens -= 1

            result = model(next_ix[None].to(device), past_key_values=past)
            next_logits, past = result['logits'][0, :], result['past_key_values']
    return tokens



if __name__ == '__main__':
    args = get_args()

    # TODO: move to config.py
    device = torch.device('cuda' if torch.cuda.is_available() and args.use_cuda else 'cpu')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2', add_prefix_space=True)
    model = GPT2LMHeadModel.from_pretrained('gpt2').train(False).to(device)

    try:
        start_tokens = tokenizer.encode(args.initial)

        tokens = generate(args.abbrev, start_tokens, model, tokenizer,
                          device=device, p_threshold=args.threshold, max_nostarting_token=args.max_tokens, temperature=args.temperature)
        print(tokenizer.decode(tokens))
    except IncorrectInput as err:
        print(sys.argv[0] + ":", err, file=sys.stderr)
        exit(1)

