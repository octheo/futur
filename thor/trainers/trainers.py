from .base_trainer import BaseTrainer

class CNNTrainer(BaseTrainer):
    
    def __init__(self, optimizer, loss, metrics):
        super().__init__(optimizer, loss, metrics)
    
    def model_prediction(self, model, images):
        return model(images)


    
class ViTTrainer(BaseTrainer):
    
    def __init__(self, optimizer, loss, metrics):
        super().__init__(optimizer, loss, metrics)
    
    def model_prediction(self, model, images):
        return model(pixel_values=images)