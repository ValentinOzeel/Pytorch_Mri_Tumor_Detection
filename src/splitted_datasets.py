from secondary_module import ConfigLoad

config_load = ConfigLoad()
dataset = config_load.get_dataset()

class SplittedDataset(dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        
    def __len__(self):
        return len(self.subset) 
    
    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y
