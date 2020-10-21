import json


def write_json_dict(input_dict, file_name):
    with open(file_name, 'w') as f:
        json.dump(input_dict, f, indent=2)
        f.close()
