from torchvision import transforms as T
from configs.decoder_defaults import default_config, ConfigField

class TrainDecoderConfig:
    def __init__(self, config):
        self.config = self.map_config(config, default_config)

    def map_config(self, config, defaults):
        """
        Returns a dictionary containing all config options in the union of config and defaults.
        If the config value is an array, apply the default value to each element.
        If the default values dict has a value of ConfigField.REQUIRED for a key, it is required and a runtime error should be thrown if a value is not supplied from config
        """
        def _check_option(option, option_config, option_defaults):
            for key, value in option_defaults.items():
                if key not in option_config:
                    if value == ConfigField.REQUIRED:
                        raise RuntimeError("Required config value '{}' of option '{}' not supplied".format(key, option))
                    option_config[key] = value
        
        for key, value in defaults.items():
            if key not in config:
                # Then they did not pass in one of the main configs. If the default is an array or object, then we can fill it in. If is a required object, we must error
                if value == ConfigField.REQUIRED:
                    raise RuntimeError("Required config value '{}' not supplied".format(key))
                elif isinstance(value, dict):
                    config[key] = {}
                elif isinstance(value, list):
                    config[key] = [{}]
            # Config[key] is now either a dict, list of dicts, or an object that cannot be checked. 
            # If it is a list, then we need to check each element
            if isinstance(value, list):
                assert isinstance(config[key], list)
                for element in config[key]:
                    _check_option(key, element, value[0])
            elif isinstance(value, dict):
                _check_option(key, config[key], value)
            # This object does not support checking
        return config

    def get_preprocessing(self):
        """
        Takes the preprocessing dictionary and converts it to a composition of torchvision transforms
        """
        def _get_transformation(transformation_name, **kwargs):
            if transformation_name == "RandomResizedCrop":
                return T.RandomResizedCrop(**kwargs)
            elif transformation_name == "RandomHorizontalFlip":
                return T.RandomHorizontalFlip()
            elif transformation_name == "ToTensor":
                return T.ToTensor()
        
        transformations = []
        for transformation_name, transformation_kwargs in self.config["data"]["preprocessing"].items():
            if isinstance(transformation_kwargs, dict):
                transformations.append(_get_transformation(transformation_name, **transformation_kwargs))
            else:
                transformations.append(_get_transformation(transformation_name))
        return T.Compose(transformations)
    
    def __getitem__(self, key):
        return self.config[key]
