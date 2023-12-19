import yaml

CONFIG_FILE = "config.yaml"


def load_config_file():
    with open(CONFIG_FILE, "r") as f:
        config = yaml.safe_load(f)
    return config


def write_config_file(data):
    with open(CONFIG_FILE, "w") as f:
        yaml.dump(data, f)
