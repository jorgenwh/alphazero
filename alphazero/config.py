class Config(dict):
    def __getattr__(self, name):
        return self[name]


def populate_config(config: Config, **kwargs):
    for key, value in kwargs.items():
        config[key] = value