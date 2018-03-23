
import argparse

def get_parser(desc):
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        "--uniform",
        action="store_false",
        dest="importance_training",
        help="Enable uniform sampling"
    )

    return parser
