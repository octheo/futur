import typing
from abc import ABC, abstractmethod
from tqdm import tqdm

from torch.utils.data import DataLoader
import torch

from .loss import Loss
from .metrics import Metric

class BaseTrainer(ABC):
    
    def __init__(self, optimizer, loss: Loss, metrics: Metric):
        
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
    
    @abstractmethod
    def model_prediction(self, model, images):
        pass
    
    def train_one_epoch(self, model, epoch, train_dataloader, nb_classes, device, wandb_run):
        running_loss = 0.
        
        model.train()
        
        for i, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader), unit="Batch", desc=f"Train epoch {epoch}"):
            images, _, labels = batch
            images, labels = images.to(device), labels.to(device)

            self.optimizer.zero_grad()
            output_logits = self.model_prediction(model, images)
            loss = self.loss.compute_loss(output_logits, labels)
            loss.backward()

            self.optimizer.step()
            
            if wandb_run is not None:
                metrics_results = self.metrics.compute_metrics(output_logits, labels)
                wandb_log = {"train_loss": loss} | metrics_results
                wandb_run.log(wandb_log)

            running_loss += loss.item()

        return running_loss / i
    
    def evaluate(self, model, epoch, val_dataloader, nb_classes, device, wandb_run):
        running_loss = 0.
        
        model.eval()
        
        with torch.no_grad():
            for i, vdata in tqdm(enumerate(val_dataloader), total=len(val_dataloader), unit="Batch", desc=f"Val epoch {epoch}"):
                images, _, labels = vdata
                images, labels = images.to(device), labels.to(device)

                output_logits = self.model_prediction(model, images)
                loss = self.loss.compute_loss(output_logits, labels)
                
                if wandb_run is not None:
                    metrics_results = self.metrics.compute_metrics(output_logits, labels, samples_set="val")
                    wandb_log = {"val_loss": loss} | metrics_results
                    wandb_run.log(wandb_log)
                
                running_loss += loss.item()
                
        return running_loss / i
    
    def train(self, model, epochs, train_dataloader, validation_dataloader, nb_classes, device, save_path="", wandb_run=None):
        best_vloss = 1_000_000.
        
        model.to(device)
        
        for epoch in range(epochs):
            
            avg_loss = self.train_one_epoch(model, epoch, train_dataloader, nb_classes, device, wandb_run)
            avg_vloss = self.evaluate(model, epoch, validation_dataloader, nb_classes, device, wandb_run)
            
            print(f"train loss: {avg_loss}, val loss: {avg_vloss}")
                  
            if wandb_run is not None:
                wandb_log = {"avg_train_loss": avg_loss, "avg_val_loss": avg_vloss}
                wandb_run.log(wandb_log)
            
            if avg_vloss < best_vloss:
                best_vloss = avg_vloss
                if len(save_path) > 1:
                    torch.save(model.state_dict(), save_path)
