import json
import os
import os.path

def make_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def write_json_to_file(data, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
def read_data_from_json(filename):
    with open(filename, encoding='utf-8') as f: 
        j = json.load(f)
        return j