from typing import Optional

import torch

from allennlp.training.metrics.metric import Metric
from allennlp.common.checks import ConfigurationError

from myallennlp.metric.labeled_f1_measure import LabeledF1Measure
@Metric.register("iter_labeled_f1")
class IterativeLabeledF1Measure(Metric):
    """
    Computes Precision, Recall and F1 with respect to a given ``positive_label``.
    For example, for a BIO tagging scheme, you would pass the classification index of
    the tag you are interested in, resulting in the Precision, Recall and F1 score being
    calculated for this tag only.
    """
    def __init__(self, negative_label: int,negative_pred:int,selected_metrics = []) -> None:
        self._negative_label = negative_label
        self._negative_pred = negative_pred
        self.labeled_f1_scores = {}
        self.selected_metrics = selected_metrics
    def __call__(self,
                 predictions: torch.Tensor,
                 gold_labels: torch.Tensor,
                graph_mask: Optional[torch.Tensor],
                 pred_probs: torch.Tensor,
                 pred_candidates: torch.Tensor,
                 gold_pred: torch.Tensor,
                 scores:torch.Tensor = torch.zeros(1),
                 linear_scores:torch.Tensor = torch.zeros(1),
                 n_iteration:int=0):
        """
        Parameters
        ----------
        predictions : ``torch.Tensor``, required.
            A tensor of predictions of shape (batch_size, ..., num_classes).
        gold_labels : ``torch.Tensor``, required.
            A tensor of integer class label of shape (batch_size, ...). It must be the same
            shape as the ``predictions`` tensor without the ``num_classes`` dimension.
        mask: ``torch.Tensor``, optional (default = None).
            A masking tensor the same size as ``gold_labels``.
        """
        labeled_f1 = self.labeled_f1_scores.setdefault(n_iteration,LabeledF1Measure(self._negative_label ,self._negative_pred))
        labeled_f1(predictions,gold_labels,graph_mask,pred_probs,pred_candidates,gold_pred,scores,linear_scores)

    def get_metric(self, reset: bool = False,training=True):
        """
        Returns
        -------
        A tuple of the following metrics based on the accumulated count statistics:
        precision : float
        recall : float
        f1-measure : float
        """
        all_metrics = {}
   #     all_metrics["cool_down"] = self.cool_down/self._total_sentences

        if training:
            sorted_scores = []

            if -1 in self.labeled_f1_scores:
                sorted_scores.append(-1)

            if 0 in self.labeled_f1_scores:
                sorted_scores.append(0)

            if 1 in  self.labeled_f1_scores:
                sorted_scores.append(1)

        else:
            sorted_scores = sorted(self.labeled_f1_scores)
            sorted_scores = sorted_scores[:-1]

        for iterations in sorted_scores:

            metrics =  self.labeled_f1_scores[iterations].get_metric()
            for metric in metrics:
                if len(self.selected_metrics)== 0 or metric in self.selected_metrics:
                    all_metrics[metric+"_"+str(iterations)] = metrics[metric]


        sorted_scores = sorted(self.labeled_f1_scores)

        iterations =  sorted_scores[-1]

        metrics =  self.labeled_f1_scores[iterations].get_metric()
        for metric in metrics:
            if len(self.selected_metrics)== 0 or metric in self.selected_metrics:
                all_metrics[metric] = metrics[metric]
        if reset:
            self.reset()
        return all_metrics

    def reset(self):
        self.labeled_f1_scores = {}