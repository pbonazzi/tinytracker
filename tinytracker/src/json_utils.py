import json

def read_json(path):
    with open(path) as json_data_file:
        data = json.load(json_data_file)

    return data

def save_json(path, data):
    with open(path, 'w') as f:
        json.dump(data, f, indent=4, separators=(", ", ": "))

def json2string(data):
    return json.dumps(data, indent=4, separators=(", ", ": "))