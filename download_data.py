# coding=utf-8

from omogre import download_model
from argparse import ArgumentParser

parser = ArgumentParser(description="Download omogre model")

parser.add_argument("--data_path", type=str, default='None', help="omogre model direcory")
args = parser.parse_args()

if __name__ == "__main__":
    path = download_model(cache_dir=args.data_path)
    print('download_model', path)
        
    