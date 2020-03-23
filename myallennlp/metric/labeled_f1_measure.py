from typing import Optional

import torch

from allennlp.training.metrics.metric import Metric
from allennlp.common.checks import ConfigurationError


@Metric.register("labeled_f1")
class LabeledF1Measure(Metric):
    """
    Computes Precision, Recall and F1 with respect to a given ``positive_label``.
    For example, for a BIO tagging scheme, you would pass the classification index of
    the tag you are interested in, resulting in the Precision, Recall and F1 score being
    calculated for this tag only.
    """
    def __init__(self, negative_label: int,negative_pred:int)->None:
        self._negative_label = negative_label
        self._negative_pred = negative_pred
        self._true_positives = 0.0
        self._true_negatives = 0.0
        self._false_positives = 0.0
        self._false_negatives = 0.0


        self._un_true_positives = 0.0
        self._un_true_negatives = 0.0
        self._un_false_positives = 0.0
        self._un_false_negatives = 0.0


        self._pred_true_positives = 0.0
        self._pred_true_negatives = 0.0
        self._pred_false_positives = 0.0
        self._pred_false_negatives = 0.0

        self.scores = 0.0
        self.linear_scores = 0.0
        self.score_size = 0.0

    def __call__(self,
                 predictions: torch.Tensor,
                 gold_labels: torch.Tensor,
                 mask: Optional[torch.Tensor] ,
                 pred_probs: torch.Tensor,
                 pred_candidates: torch.Tensor,
                 gold_pred: torch.Tensor,
                 scores:torch.Tensor,
                 linear_scores:torch.Tensor):
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
        predictions, gold_labels, mask, pred_probs,pred_candidates,gold_pred,scores,linear_scores = self.unwrap_to_tensors(predictions, gold_labels, mask,pred_probs,pred_candidates,gold_pred,scores,linear_scores)

        self.score_size += linear_scores.view(-1).size(0)
        self.scores += (scores*mask.unsqueeze(-1)).sum().item() if scores is not None else 0
        self.linear_scores += (linear_scores*mask.unsqueeze(-1)).sum().item()
        num_classes = predictions.size(-1)
        if (gold_labels >= num_classes).any():
            raise ConfigurationError("A gold label passed to F1Measure contains an id >= {}, "
                                     "the number of classes.".format(num_classes))
        mask = mask.float()
        gold_labels = gold_labels.float()

        negative_label_mask = gold_labels.eq(self._negative_label).float()
        positive_label_mask = 1.0 - negative_label_mask

        assert predictions.dim() == gold_labels.dim()+1,(predictions.size(),gold_labels.size())
        argmax_predictions = predictions.argmax(-1).float()


        assert argmax_predictions.size() == negative_label_mask.size(),(predictions.size(),gold_labels.size(),argmax_predictions.size(),negative_label_mask.size())
        assert argmax_predictions.size() == gold_labels.size(),(argmax_predictions.size(),gold_labels.size())
        assert negative_label_mask.size() == mask.size(),(negative_label_mask.size(),mask.size())
        # True Negatives: correct non-positive predictions.
        correct_null_predictions = (argmax_predictions ==
                                    gold_labels).float() * negative_label_mask
        self._true_negatives += (correct_null_predictions.float() * mask).sum()

        # True Positives: correct positively labeled predictions.
        correct_non_null_predictions = (argmax_predictions ==
                                        gold_labels).float() * positive_label_mask
        self._true_positives += (correct_non_null_predictions * mask).sum()

        # False Negatives: incorrect negatively labeled predictions.
        incorrect_null_predictions = (argmax_predictions !=
                                      gold_labels).float() * positive_label_mask
        self._false_negatives += (incorrect_null_predictions * mask).sum()

        # False Positives: incorrect positively labeled predictions
        incorrect_non_null_predictions = (argmax_predictions !=
                                          gold_labels).float() * negative_label_mask
        self._false_positives += (incorrect_non_null_predictions * mask).sum()




        argmax_predictions = argmax_predictions > 0
        gold_labels = gold_labels > 0
        # True Negatives: correct non-positive predictions.
        correct_null_predictions = (argmax_predictions ==
                                    gold_labels).float() * negative_label_mask
        self._un_true_negatives += (correct_null_predictions.float() * mask).sum()

        # True Positives: correct positively labeled predictions.
        correct_non_null_predictions = (argmax_predictions ==
                                        gold_labels).float() * positive_label_mask
        self._un_true_positives += (correct_non_null_predictions * mask).sum()

        # False Negatives: incorrect negatively labeled predictions.
        incorrect_null_predictions = (argmax_predictions !=
                                      gold_labels).float() * positive_label_mask
        self._un_false_negatives += (incorrect_null_predictions * mask).sum()

        # False Positives: incorrect positively labeled predictions
        incorrect_non_null_predictions = (argmax_predictions !=
                                          gold_labels).float() * negative_label_mask
        self._un_false_positives += (incorrect_non_null_predictions * mask).sum()




        pred_mask = (gold_pred>0).float()
        negative_label_mask = gold_pred.eq(self._negative_pred).float()
        positive_label_mask = 1.0 - negative_label_mask


        # (batch_size, length, 1) index to pred_candidates
        predindex_argmax =  pred_probs.argmax(-1).unsqueeze(-1)

        # (batch_size, length) pred_id

        pred_argmax = pred_candidates.gather(-1,predindex_argmax).squeeze(-1)


        # True Negatives: correct non-positive predictions.
        correct_null_predictions = (pred_argmax ==
                                    gold_pred).float() * negative_label_mask
        self._pred_true_negatives += (correct_null_predictions.float() *pred_mask).sum()

        # True Positives: correct positively labeled predictions.
        correct_non_null_predictions = (pred_argmax ==
                                        gold_pred).float() * positive_label_mask
        self._pred_true_positives += (correct_non_null_predictions *pred_mask).sum()

        # False Negatives: incorrect negatively labeled predictions.
        incorrect_null_predictions = (pred_argmax !=
                                      gold_pred).float() * positive_label_mask
        self._pred_false_negatives += (incorrect_null_predictions *pred_mask).sum()

        # False Positives: incorrect positively labeled predictions
        incorrect_non_null_predictions = (pred_argmax !=
                                          gold_pred).float() * negative_label_mask
        self._pred_false_positives += (incorrect_non_null_predictions *pred_mask).sum()



    def get_metric(self, reset: bool = False):
        """
        Returns
        -------
        A tuple of the following metrics based on the accumulated count statistics:
        precision : float
        recall : float
        f1-measure : float
        """
        _true_positives = self._true_positives + self._pred_true_positives
        _false_positives = self._false_positives + self._pred_false_positives
        _false_negatives = self._false_negatives + self._pred_false_negatives

        precision = float(_true_positives) / float(_true_positives + _false_positives + 1e-13)
        recall = float(_true_positives) / float(_true_positives + _false_negatives + 1e-13)
        f1_measure = 2. * ((precision * recall) / (precision + recall + 1e-13))



        label_precision = float(self._true_positives) / float(self._true_positives + self._false_positives + 1e-13)
        label_recall = float(self._true_positives) / float(self._true_positives + self._false_negatives + 1e-13)
        label_f1_measure = 2. * ((label_precision * label_recall) / (label_precision + label_recall + 1e-13))


        pred_precision = float(self._pred_true_positives) / float(self._pred_true_positives + self._pred_false_positives + 1e-13)
        pred_recall = float(self._pred_true_positives) / float(self._pred_true_positives + self._pred_false_negatives + 1e-13)
        pred_f1_measure = 2. * ((pred_precision * pred_recall) / (pred_precision + pred_recall + 1e-13))



        un_precision = float(self._un_true_positives) / float(self._un_true_positives + self._un_false_positives + 1e-13)
        un_recall = float(self._un_true_positives) / float(self._un_true_positives + self._un_false_negatives + 1e-13)
        un_f1_measure = 2. * ((un_precision * un_recall) / (un_precision + un_recall + 1e-13))
        if reset:
            self.reset()
        def format(number):
            return number
        metrics = {}
        metrics["u_P"] = format(un_precision)
        metrics["u_R"] =   format(un_recall)

        metrics["u_F"] =   format(un_f1_measure)
        metrics["p_F"] =   format(pred_f1_measure)
        metrics["l_F"] =   format(label_f1_measure)
        metrics["l_P"] =   format(label_precision)
        metrics["l_R"] =   format(label_recall)

        metrics["P"] =  format(precision)
        metrics["R"] =   format(recall)
        metrics["F"] =  format(f1_measure)
        metrics["h_S"] =  format(self.scores/self.score_size)
        metrics["l_S"] =  format(self.linear_scores/self.score_size)
        return metrics

    def reset(self):
        self._true_positives = 0.0
        self._true_negatives = 0.0
        self._false_positives = 0.0
        self._false_negatives = 0.0

        self._un_true_positives = 0.0
        self._un_true_negatives = 0.0
        self._un_false_positives = 0.0
        self._un_false_negatives = 0.0


        self._pred_true_positives = 0.0
        self._pred_true_negatives = 0.0
        self._pred_false_positives = 0.0
        self._pred_false_negatives = 0.0

        self.scores = 0.0
        self.linear_scores = 0.0
        self.score_size = 0.0