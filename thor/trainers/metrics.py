import typing
from abc import ABC, abstractmethod

from torcheval.metrics import MulticlassRecall, MulticlassPrecision, MulticlassF1Score, MulticlassAUPRC
import torch

class Metric(ABC):
    
    def __init__(self, selected_metrics: list[str]):
        self.selected_metrics = selected_metrics
    
    @abstractmethod
    def compute_metrics(self, pred, labels):
        pass

        
class ClassificationMetrics(Metric):
    
    def __init__(self, nb_classes: int, selected_metrics: list[str]):
        super().__init__(selected_metrics)
        self.nb_classes = nb_classes
        self.cls_mapping = {"f1": MulticlassF1Score(average="macro", num_classes=self.nb_classes),
                                 "precision": MulticlassPrecision(average="macro", num_classes=self.nb_classes),
                                 "recall": MulticlassRecall(average="macro", num_classes=self.nb_classes)
                                 "AP": MulticlassAUPRC(num_classes=self.nb_classes)
                                } 
    
    def compute_metrics(self, dataloader, output_logits, labels, samples_set="train"):
        computed_metrics = {}
        _, predicted = torch.max(output_logits, 1)
        for metric in self.selected_metrics:
            for batch in dataloader:
                images, _, labels = batch
                metric = self.function_mapping[metric]
                metric.update(output_logits, labels)
            result = metric.compute()
            metric.reset()
            computed_metrics[samples_set + "_" + metric] = result
        
        return computed_metrics