"""
A :class:`~allennlp.data.dataset_readers.dataset_reader.DatasetReader`
reads a file and converts it to a collection of
:class:`~allennlp.data.instance.Instance` s.
The various subclasses know how to read specific filetypes
and produce datasets in the formats required by specific models.
"""

# pylint: disable=line-too-long

from myallennlp.dataset_readers.MultiCandidatesSequence import MultiCandidatesSequence
from myallennlp.dataset_readers.multiindex_field import MultiIndexField
from myallennlp.dataset_readers.nonsquare_adjacency_field import NonSquareAdjacencyField
from myallennlp.dataset_readers.index_sequence_label_field import IndexSequenceLabelField