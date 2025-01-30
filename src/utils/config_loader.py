import yaml


def load_config(config_path, key=None):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    if key:
        return config[key]
    return config