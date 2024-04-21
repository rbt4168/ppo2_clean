import os
import torch

def ensure_dir(file_path):
    '''
    Used to ensure to create the a directory when needed
    :param file_path: path to the file that we want to create
    '''
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    return file_path

def save_checkpoint(state, path, filename='checkpoint.pth.tar', version=0):
    torch.save(state, ensure_dir(os.path.join(path, version, filename)))

def load_checkpoint(path, filename='checkpoint.pth.tar', version=0):
    return torch.load(ensure_dir(os.path.join(path, version, filename)))