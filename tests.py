# TODO: run this code somehow
test_tokens = [tokenizer.decode(ix) for ix in tokenizer.encode('star-like')]
assert test_tokens == [' star', '-', 'like'], "Settings of tokenizer have changed"

assert list(map(is_starting, test_tokens)) == [True, False, False]
assert list(map(is_continuing, test_tokens)) == [False, True, True]
assert not is_starting('\n') and not is_continuing('\n')
del test_tokens
