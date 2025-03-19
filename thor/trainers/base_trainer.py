import typing
from abc import ABC, abstractmethod
from tqdm import tqdm

from torch.utils.data import DataLoader
import torch
import wandb

from .loss import Loss
from .metrics import Metric

class BaseTrainer(ABC):
    
    def __init__(self, optimizer, loss: Loss, metrics: Metric):
        
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
    
    
    def model_prediction(self, model, images):
        return model(images)
    
    @abstractmethod
    def train_one_epoch(self, model, epoch, train_dataloader, nb_classes, device, wandb_run):
        pass

    @abstractmethod
    def log_image_table(self, images, predicted, labels, nb_classes, wandb_run, probs):
        pass

    @abstractmethod
    def validate_model(self, model, test_dl, nb_classes, device, wandb_run, log_images=False, batch_idx=0):
        pass

    @abstractmethod
    def evaluate(self, model, epoch, val_dataloader, nb_classes, device, wandb_run):
        pass
    
    def train(self, model, epochs, train_dataloader, validation_dataloader, nb_classes, device, save_path="", wandb_run=None, model_artifact=None):
        best_vloss = 1_000_000.
        
        model.to(device)
        
        for epoch in range(epochs):
            
            avg_loss, metric_results = self.train_one_epoch(model, epoch, train_dataloader, nb_classes, device, wandb_run)
            avg_vloss, vmetric_results = self.evaluate(model, epoch, validation_dataloader, nb_classes, device, wandb_run)
            
            print(f"train loss: {avg_loss}, val loss: {avg_vloss}")
                  
            if wandb_run is not None:
                wandb_log = metric_results | vmetric_results | {"avg_train_loss": avg_loss, "avg_val_loss": avg_vloss, "epoch": epoch}
                wandb_run.log(wandb_log)
            
            if avg_vloss < best_vloss:
                best_vloss = avg_vloss
                if len(save_path) > 1:
                    torch.save(model.state_dict(), save_path)
            
        if model_artifact is not None:
            model_artifact.add_file(save_path)
            wandb.save(save_path)
        
        wandb_run.log_artifact(model_artifact)
