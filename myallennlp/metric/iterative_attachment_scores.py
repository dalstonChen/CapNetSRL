from typing import Optional, List

from overrides import overrides
import torch

from allennlp.training.metrics.metric import Metric
from allennlp.training.metrics.attachment_scores import AttachmentScores


@Metric.register("iterative_attachment_scores")
class IterativeAttachmentScores(Metric):
    """
    Computes labeled and unlabeled attachment scores for a
    dependency parse, as well as sentence level exact match
    for both labeled and unlabeled trees. Note that the input
    to this metric is the sampled predictions, not the distribution
    itself.

    Parameters
    ----------
    ignore_classes : ``List[int]``, optional (default = None)
        A list of label ids to ignore when computing metrics.
    """
    def __init__(self, ignore_classes: List[int] = None) -> None:

        self._ignore_classes: List[int] = ignore_classes or []
        self._attachment_scores = {}
        self._total_sentences = 0.
        self.refine_lr = 0
    def __call__(self, # type: ignore
                 predicted_indices: torch.Tensor,
                 predicted_labels: torch.Tensor,
                 gold_indices: torch.Tensor,
                 gold_labels: torch.Tensor,
                 mask: Optional[torch.Tensor] = None,
                 n_iteration:int = 0,
                 refine_lr:float=0):
        """
        Parameters
        ----------
        predicted_indices : ``torch.Tensor``, required.
            A tensor of head index predictions of shape (batch_size, timesteps).
        predicted_labels : ``torch.Tensor``, required.
            A tensor of arc label predictions of shape (batch_size, timesteps).
        gold_indices : ``torch.Tensor``, required.
            A tensor of the same shape as ``predicted_indices``.
        gold_labels : ``torch.Tensor``, required.
            A tensor of the same shape as ``predicted_labels``.
        mask: ``torch.Tensor``, optional (default = None).
            A tensor of the same shape as ``predicted_indices``.
        """

        self._total_sentences += gold_indices.size(0)
        self.refine_lr += refine_lr * gold_indices.size(0)
        attachment_scores = self._attachment_scores.setdefault(n_iteration,AttachmentScores(self._ignore_classes))

        attachment_scores(predicted_indices,predicted_labels,gold_indices,gold_labels,mask=mask)

    def get_metric(self, reset: bool = False,training=True):
        """
        Returns
        -------
        The accumulated metrics as a dictionary.
        """

        all_metrics = {}
        all_metrics["refine_lr"] = self.refine_lr/self._total_sentences
   #     all_metrics["cool_down"] = self.cool_down/self._total_sentences
        sorted_scores = sorted(self._attachment_scores)
        if training:
            sorted_scores = [sorted_scores[0]] if len(sorted_scores) > 1 else []
        else:
            sorted_scores = sorted_scores[:-1]

        for iterations in sorted_scores:

            metrics =  self._attachment_scores[iterations].get_metric()
            for metric in metrics:
                all_metrics[metric+str(iterations)] = metrics[metric]

        iterations =  len(self._attachment_scores)-1

        metrics =  self._attachment_scores[iterations].get_metric()
        for metric in metrics:
            all_metrics[metric] = metrics[metric]
        if reset:
            self.reset()
        return all_metrics

    @overrides
    def reset(self):
        self._attachment_scores = {}
        self._total_sentences = 0.
        self.refine_lr = 0
        self.cool_down = 0
