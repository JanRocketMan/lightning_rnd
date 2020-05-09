import yaml

with open('config.yaml') as f:
    default_config = yaml.load(f, Loader=yaml.FullLoader)
