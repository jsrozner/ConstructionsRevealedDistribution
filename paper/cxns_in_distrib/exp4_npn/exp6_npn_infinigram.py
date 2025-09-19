# todo: prob don't need this file
import json

def save_dict_to_file(d: dict[str, int], filepath: str):
    with open(filepath, 'w') as f:
        json.dump(d, f)

def load_dict_from_file(filepath: str) -> dict[str, int]:
    with open(filepath, 'r') as f:
        return json.load(f)