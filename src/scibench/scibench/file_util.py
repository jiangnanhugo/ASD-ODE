import os
import pickle
import sys
from pathlib import Path
import yaml
import json
from cryptography.fernet import Fernet
from sympy.parsing.sympy_parser import parse_expr


def decrypt_equation(eq_file, key_filename):
    with open(key_filename, 'rb') as filekey:
        key = filekey.read()
    fernet = Fernet(key)
    with open(eq_file, 'rb') as enc_file:
        encrypted = enc_file.read()

    decrypted = fernet.decrypt(encrypted)
    one_equation = json.loads(decrypted)
    one_equation['eq_expression'] = parse_expr(one_equation['eq_expression'])
    print("-" * 20)
    for key in one_equation:
        print(key, "\t", one_equation[key])
    print("-" * 20)
    return one_equation


def check_if_exists(file_path):
    return file_path is not None and os.path.exists(file_path)



def is_float(s):
    """Determine whether the input variable can be cast to float."""
    try:
        float(s)
        return True
    except ValueError:
        return False


def get_file_path_list(dir_path, is_recursive=False, is_sorted=False):
    file_list = list()
    for file in os.listdir(dir_path):
        path = os.path.join(dir_path, file)
        if os.path.isfile(path):
            file_list.append(path)
        elif is_recursive:
            file_list.extend(get_file_path_list(path, is_recursive))
    return sorted(file_list) if is_sorted else file_list


def get_dir_path_list(dir_path, is_recursive=False, is_sorted=False):
    dir_list = list()
    for file in os.listdir(dir_path):
        path = os.path.join(dir_path, file)
        if os.path.isdir(path):
            dir_list.append(path)
        elif is_recursive:
            dir_list.extend(get_dir_path_list(path, is_recursive))
    return sorted(dir_list) if is_sorted else dir_list


def make_dirs(dir_path):
    Path(dir_path).mkdir(parents=True, exist_ok=True)


def make_parent_dirs(file_path):
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)


def save_pickle(entity, file_path):
    make_parent_dirs(file_path)
    with open(file_path, 'wb') as fp:
        pickle.dump(entity, fp)


def load_pickle(file_path):
    with open(file_path, 'rb') as fp:
        return pickle.load(fp)


def get_binary_object_size(x, unit_size=1024):
    return sys.getsizeof(pickle.dumps(x)) / unit_size


def yaml_join(loader, node):
    seq = loader.construct_sequence(node)
    return ''.join([str(i) for i in seq])


def yaml_pathjoin(loader, node):
    seq = loader.construct_sequence(node)
    return os.path.expanduser(os.path.join(*[str(i) for i in seq]))


def load_yaml_file(yaml_file_path, custom_mode=True):
    if custom_mode:
        yaml.add_constructor('!join', yaml_join, Loader=yaml.FullLoader)
        yaml.add_constructor('!pathjoin', yaml_pathjoin, Loader=yaml.FullLoader)
    with open(yaml_file_path, 'r') as fp:
        return yaml.load(fp, Loader=yaml.FullLoader)
