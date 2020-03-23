from typing import Dict, Union, Sequence, Set, Optional, cast, List
import logging

from overrides import overrides
import torch

from allennlp.data.fields.field import Field
from allennlp.data.fields.sequence_field import SequenceField
from allennlp.data.vocabulary import Vocabulary
from allennlp.common.checks import ConfigurationError

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class MultiCandidatesSequence(Field[torch.Tensor]):
    """
    A ``MultiLabelField`` is an extension of the :class:`LabelField` that allows for multiple labels.
    It is particularly useful in multi-label classification where more than one label can be correct.
    As with the :class:`LabelField`, labels are either strings of text or 0-indexed integers (if you wish
    to skip indexing by passing skip_indexing=True).
    If the labels need indexing, we will use a :class:`Vocabulary` to convert the string labels
    into integers.

    This field will get converted into a vector of length equal to the vocabulary size with
    one hot encoding for the labels (all zeros, and ones for the labels).

    Parameters
    ----------
    labels : ``Sequence[Union[str, int]]``
    label_namespace : ``str``, optional (default="labels")
        The namespace to use for converting label strings into integers.  We map label strings to
        integers for you (e.g., "entailment" and "contradiction" get converted to 0, 1, ...),
        and this namespace tells the ``Vocabulary`` object which mapping from strings to integers
        to use (so "entailment" as a label doesn't get the same integer id as "entailment" as a
        word).  If you have multiple different label fields in your data, you should make sure you
        use different namespaces for each one, always using the suffix "labels" (e.g.,
        "passage_labels" and "question_labels").
    skip_indexing : ``bool``, optional (default=False)
        If your labels are 0-indexed integers, you can pass in this flag, and we'll skip the indexing
        step.  If this is ``False`` and your labels are not strings, this throws a ``ConfigurationError``.
    num_labels : ``int``, optional (default=None)
        If ``skip_indexing=True``, the total number of possible labels should be provided, which is required
        to decide the size of the output tensor. `num_labels` should equal largest label id + 1.
        If ``skip_indexing=False``, `num_labels` is not required.

    """
    # It is possible that users want to use this field with a namespace which uses OOV/PAD tokens.
    # This warning will be repeated for every instantiation of this class (i.e for every data
    # instance), spewing a lot of warnings so this class variable is used to only log a single
    # warning per namespace.
    _already_warned_namespaces: Set[str] = set()

    def __init__(self,
                 labels: Sequence[List[Union[str, int]]],
                 label_namespace: str = 'pred',
                 skip_indexing: bool = False,
                 num_labels: Optional[int] = None) -> None:
        self.labels = labels
        self._label_namespace = label_namespace
        self._label_ids = None
        self._maybe_warn_for_namespace(label_namespace)
        max_senses = max([len(labels) for labels in self.labels]+[1])
        self._num_labels = max_senses

        for candidates in labels:
            if len(candidates) > self._num_labels:
                self._num_labels = len(candidates)
        if skip_indexing:
            if not all(isinstance(label, int) for label in labels):
                raise ConfigurationError("In order to skip indexing, your labels must be integers. "
                                         "Found labels = {}".format(labels))
            if not num_labels:
                raise ConfigurationError("In order to skip indexing, num_labels can't be None.")

            if not all(cast(int, label) < num_labels for label in labels):
                raise ConfigurationError("All labels should be < num_labels. "
                                         "Found num_labels = {} and labels = {} ".format(num_labels, labels))

            self._label_ids = labels
        else:
            if not all(isinstance(label, List) for label in labels):
                raise ConfigurationError("MultiLabelFields expects list of string labels if skip_indexing=False. "
                                         "Found labels: {}".format(labels))

    def _maybe_warn_for_namespace(self, label_namespace: str) -> None:
        if not (label_namespace.endswith("labels") or label_namespace.endswith("tags")):
            if label_namespace not in self._already_warned_namespaces:
                logger.warning("Your label namespace was '%s'. We recommend you use a namespace "
                               "ending with 'labels' or 'tags', so we don't add UNK and PAD tokens by "
                               "default to your vocabulary.  See documentation for "
                               "`non_padded_namespaces` parameter in Vocabulary.",
                               self._label_namespace)
                self._already_warned_namespaces.add(label_namespace)

    @overrides
    def count_vocab_items(self, counter: Dict[str, Dict[str, int]]):
        if self._label_ids is None:
            for labels in self.labels:
                for label in labels:
                    counter[self._label_namespace][label] += 1  # type: ignore

    @overrides
    def index(self, vocab: Vocabulary):
        if self._label_ids is None:
            self._label_ids = [[vocab.get_token_index(label, self._label_namespace)  # type: ignore
                               for label in labels] for labels in self.labels]

    @overrides
    def get_padding_lengths(self) -> Dict[str, int]:  # pylint: disable=no-self-use

        max_senses = max([len(labels) for labels in self.labels]+[0])
        self._num_labels = max_senses

        return { "num_heads": len(self.labels),"max_senses":max_senses}

    @overrides
    def as_tensor(self, padding_lengths: Dict[str, int]) -> torch.Tensor:
        # pylint: disable=unused-argument

        num_heads = padding_lengths['num_heads']  #predicate
        max_senses  = padding_lengths['max_senses']

        tensor = torch.zeros(num_heads,max_senses).long()  # vector of zeros
        if self._label_ids:
            for i in range(len(self._label_ids)):
                if len(self._label_ids[i]) > 0:
                    data_t = torch.LongTensor(self._label_ids[i])
                    data_length = data_t.size(0)
                    tensor[i].narrow(0, 0, data_length).copy_(data_t)

        return tensor

    @overrides
    def empty_field(self):
        return MultiCandidatesSequence([[]], self._label_namespace, skip_indexing=True)

    def __str__(self) -> str:
        return f"MultiCandidatesSequence with labels: {self.labels} in namespace: '{self._label_namespace}'  max_senses:: '{self._num_labels}'.'"
