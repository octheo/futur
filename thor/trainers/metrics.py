import typing
from abc import ABC, abstractmethod

from torcheval.metrics.functional import multiclass_f1_score, multiclass_precision, multiclass_recall
from torcheval.metrics import MulticlassAUPRC
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
        self.function_mapping = {"f1": multiclass_f1_score, 
                                 "precision": multiclass_precision,
                                 "recall": multiclass_recall,
                                 "AP": self.AP
                                }
    
    def compute_metrics(self, output_logits, labels, samples_set="train"):
        computed_metrics = {}
        _, predicted = torch.max(output_logits, 1)
        for metric in self.selected_metrics:
            metric_fn = self.function_mapping[metric]
            result = metric_fn(output_logits, labels, num_classes=self.nb_classes)
            computed_metrics[samples_set + "_" + metric] = result
        
        return computed_metrics
    
    def AP(self, output_logits, labels, num_classes=None):
        average_precision = MulticlassAUPRC(num_classes=num_classes)
        average_precision.update(output_logits, labels)
        return average_precision.compute()