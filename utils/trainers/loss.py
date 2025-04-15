import typing

import torch.nn as nn

class Loss():
    
    def __init__(self, selected_loss):
        self.selected_loss = selected_loss
    
    def compute_loss(self, pred, labels):
        pass

        
class ClassificationLoss(Loss):
    
    # CE -> Cross-Entropy
    def __init__(self, selected_loss: str):
        super().__init__(selected_loss)
        
        if self.selected_loss == "CE":
            self.compute_loss = nn.CrossEntropyLoss()
        else:
            print("Select a valid loss")
    
    