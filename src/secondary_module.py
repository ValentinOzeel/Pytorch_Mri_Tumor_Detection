import os 
import yaml
from typing import List
from torchvision import transforms

# Assuming data_exploration.py is in src\main.py
project_root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
config_path = os.path.join(project_root_path, 'conf', 'config.yml')


class ConfigLoad():
    def __init__(self, path=config_path):
        self.path = path
        with open(self.path, 'r') as file:
            self.config = yaml.safe_load(file)
            
    def get_config(self):
        return self.config
            
    def get_transform(self, dict_name='DATA_TRANSFORM_AND_AUGMENTATION') -> List:
        '''
        Access transformation dict defined in config
        Transform it as a list of torchvision.transforms steps
        '''
        yml_dict = self.config[dict_name]
        steps = []
        for step_name, params in yml_dict.items():
            # Get the transforms method
            transform_step = getattr(transforms, step_name)
            # Initialize the transform method with its defined parameters and append in list
            if params: 
                steps.append(transform_step(**params))
            else:
                steps.append(transform_step()) 
        return steps