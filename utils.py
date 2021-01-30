import os


def get_dataset_path(filename):
    dirname, _ = os.path.split(os.path.abspath(__file__))
    dataset_name = dirname + '\\' + filename
    return dataset_name
