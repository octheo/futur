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
            if wandb_run is not None:
                wandb_run.log({"train_loss": loss})
            self.optimizer.step()
            
        if wandb_run is not None:
            metrics_results = self.metrics.compute_metrics(self, model, train_dataloader, device)
            wandb_run.log(metrics_results)

            running_loss += loss.item()

        return running_loss / i
    
    def log_image_table(self, images, predicted, labels, nb_classes, wandb_run, probs):
        # Create a wandb Table to log images, labels and predictions to
        table = wandb.Table(
            columns=["image", "pred", "target"] + [f"score_{i}" for i in range(nb_classes)]
        )
        for img, pred, targ, prob in zip(
            images.to("cpu"), predicted.to("cpu"), labels.to("cpu"), probs.to("cpu")
        ):
            table.add_data(wandb.Image(img[0].numpy() * 255), pred, targ, *prob.numpy())
        wandb_run.log({"predictions_table": table}, commit=False)

    def validate_model(self, model, test_dl, nb_classes, device, wandb_run, log_images=False, batch_idx=0):
        model.eval()
        val_loss = 0.0
        with torch.inference_mode():
            correct = 0
            for i, (images, _, labels) in enumerate(test_dl):
                images, labels = images.to(device), labels.to(device)

                output_logits = self.model_prediction(model, images)
                loss = self.loss.compute_loss(output_logits, labels)

                if i == batch_idx and log_images:
                    self.log_image_table(images, output_logits, labels, nb_classes, wandb_run, output_logits.softmax(dim=1))

            metrics_results = self.metrics.compute_metrics(self, model, test_dl, device, sample_set="test")
            for k, v in metrics_results.items():
                wandb_run.summary[f"test_{k}"] = v

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
                    wandb_run.log({"val_loss": loss})

            if wandb_run is not None:
                metrics_results = self.metrics.compute_metrics(self, model, val_dataloader, device, samples_set="val")
                wandb_run.log(metrics_results)
                
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
