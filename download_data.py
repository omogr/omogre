# coding=utf-8

from omogre import find_model
from argparse import ArgumentParser

parser = ArgumentParser(description="Download omogre model")

parser.add_argument("--data_path", type=str, default='None', help="omogre model direcory")
parser.add_argument("--file_name", type=str, default='accentuator_transcriptor_tiny', help="omogre model direcory")
args = parser.parse_args()

if __name__ == "__main__":
    path = find_model(file_name=file_name, cache_dir=args.data_path)
    print('find_model', path)
        
    