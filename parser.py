import argparse

def get_args(args=None):
    parser = argparse.ArgumentParser(prog='denotarikon')
    parser.add_argument('abbrev', help='abbreviature to explain')
    parser.add_argument('initial', nargs='?', default='', help='start generation from this phrase')

    model_group = parser.add_argument_group(title='Model', description='Settings affecting the text generation process')
    model_group.add_argument('--threshold', type=float, default=0.95, metavar='P', help='Probability threshold for nucleus sampling')
    model_group.add_argument('--temperature', type=float, default=1.0, metavar='TEMP')
    model_group.add_argument('--max-tokens', type=int, default=5, help='Maximum number of *tokens* allowed per one letter')

    device_group = parser.add_argument_group(title='Devices', description='Settings affectnig the devices used for text generation')
    device_group.add_argument('--no-cuda', '--no-gpu', default=True, action='store_false', help='Do not use GPU', dest='use_cuda')

    return parser.parse_args(args)


if __name__ == '__main__':
    # Debug mode: just show parsed arguments.
    print(get_args())

