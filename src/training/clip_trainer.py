import torch.optim as optim
import torch.nn as nn
from utils.misc import load_config
import tqdm


class ClipTrainer:
    def __init__(self, config_path: str = 'trainer_config.yaml') -> None:
        self.CFG = load_config(config_path)
    
    def __init_train_dependencies(self):
        self.criterion = nn.CrossEntropyLoss() if self.CFG['Head']['out_dim'] > 1 else nn.BCEWithLogitsLoss()
        if self.CFG['train']['optimizer_name'] == 'Adam':
            optim.Adam(self.model.parameters(), lr=self.CFG['train']['optimizer']['Adam']['lr'], 
                       betas=self.CFG['train']['optimizer']['Adam']['betas'])
            
            

    def fit(self):
        raise NotImplementedError()


        

    