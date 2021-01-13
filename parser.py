import argparse
from functools import wraps
from numbers import Number
from typing import List, Optional, Type, TypeVar

T = TypeVar('T', bound=Number)
def bounded_number(nmin : Optional[T] = None, nmax : Optional[T] = None, ntype : Type[T] = int) -> T:
    @wraps(ntype)
    def parse(value):
        value = ntype(value)
        if nmin is not None and nmin > value:
            raise argparse.ArgumentTypeError(f"Invalid number: should be at least {nmin}, got {value}")
        if nmax is not None and nmax < value:
            raise argparse.ArgumentTypeError(f"Invalid number: should be at most {nmax}, got {value}")
        return value
    return parse


def get_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog='denotarikon')
    parser.add_argument('abbrev', help='abbreviature to explain')
    parser.add_argument('initial', nargs='?', default='', help='start generation from this phrase')

    model_group = parser.add_argument_group(title='Model', description='Settings affecting the text generation process')
    model_group.add_argument('-n', '--number', type=bounded_number(1), default=1, help='Number of candidates to generate', dest='repeats')
    model_group.add_argument('--threshold', type=bounded_number(0., 1., ntype=float), default=0.95, metavar='P', help='Probability threshold for nucleus sampling')
    model_group.add_argument('--temperature', type=bounded_number(0., ntype=float), default=1.0, metavar='TEMP')
    model_group.add_argument('--max-tokens', type=bounded_number(0), default=5, help='Maximum number of *tokens* allowed per one letter')

    device_group = parser.add_argument_group(title='Devices', description='Settings affectnig the devices used for text generation')
    device_group.add_argument('--no-cuda', '--no-gpu', default=True, action='store_false', help='Do not use GPU', dest='use_cuda')
    device_group.add_argument('--directory', type=str, default=None, help='Path to a directory containing model weights or where weights should be saved. If not provided, weights are not cached')

    return parser.parse_args(args)


if __name__ == '__main__':
    # Debug mode: just show parsed arguments.
    print(get_args())

