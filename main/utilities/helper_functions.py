import os


def create_folder(path):
    existed = True
    if not os.path.exists(path):
        os.makedirs(path)
        existed = False
    return existed