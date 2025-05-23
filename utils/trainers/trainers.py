from tqdm import tqdm

import torch
import wandb

from .base_trainer import BaseTrainer

class ClassificationTrainer(BaseTrainer):
    
    def __init__(self, optimizer, loss, metrics):
        super().__init__(optimizer, loss, metrics)
    
    def train_one_epoch(self, model, epoch, train_dataloader, nb_classes, device, wandb_run):
        running_loss = 0.
        
        model.train()
        
        for i, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader), unit="Batch", desc=f"Train epoch {epoch}"):
            images, labels = batch
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
            running_loss += loss.item()

            return running_loss / i, metrics_results

        return running_loss / i, None
    
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
            for i, (images, labels) in enumerate(test_dl):
                images, labels = images.to(device), labels.to(device)

                output_logits = self.model_prediction(model, images)
                _, predicted = torch.max(output_logits, 1)
                loss = self.loss.compute_loss(output_logits, labels)

                if i == batch_idx and log_images:
                    self.log_image_table(images, predicted, labels, nb_classes, wandb_run, output_logits.softmax(dim=1))

            metrics_results = self.metrics.compute_metrics(self, model, test_dl, device, samples_set="test")
            for k, v in metrics_results.items():
                wandb_run.summary[f"test_{k}"] = v

    def evaluate(self, model, epoch, val_dataloader, nb_classes, device, wandb_run):
        running_loss = 0.
        
        model.eval()
        
        with torch.no_grad():
            for i, vdata in tqdm(enumerate(val_dataloader), total=len(val_dataloader), unit="Batch", desc=f"Val epoch {epoch}"):
                images, labels = vdata
                images, labels = images.to(device), labels.to(device)

                output_logits = self.model_prediction(model, images)
                loss = self.loss.compute_loss(output_logits, labels)
                if wandb_run is not None:
                    wandb_run.log({"val_loss": loss})

            if wandb_run is not None:
                metrics_results = self.metrics.compute_metrics(self, model, val_dataloader, device, samples_set="val")
                running_loss += loss.item()

                return running_loss / i, metrics_results
                
        return running_loss / i, None
