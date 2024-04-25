import yaml

def read_yaml(path):
    with open(path, 'r') as file:
        yaml_file = yaml.safe_load(file)
    return yaml_file