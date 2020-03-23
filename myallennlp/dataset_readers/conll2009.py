from typing import Dict, Tuple, List
import logging,os

#import sys
#sys.path.append("/afs/inf.ed.ac.uk/user/x/xchen13/project/SRL/")

import numpy as np
from overrides import overrides
from conllu.parser import parse_line, DEFAULT_FIELDS

from allennlp.common.tqdm import Tqdm
from collections import OrderedDict, defaultdict

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, SequenceLabelField, MetadataField,AdjacencyField,MultiLabelField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token
from myallennlp.dataset_readers.MultiCandidatesSequence import MultiCandidatesSequence
from myallennlp.dataset_readers.multiindex_field import MultiIndexField
from myallennlp.dataset_readers.nonsquare_adjacency_field import NonSquareAdjacencyField
from myallennlp.dataset_readers.index_sequence_label_field import IndexSequenceLabelField
import difflib
import xml.etree.ElementTree as ET

def folder_to_files_path(folder,ends =".txt"):
    files = os.listdir(folder)
    files_path = []
    for f in files:
        if f.endswith(ends):
            files_path.append(folder+f)
          #  break
    return files_path

class PropbankReader:
    def parse(self):
        self.frames = dict()
        for f in self.frame_files_path:
            self.parse_file(f)

    def __init__(self, folder_path):
        self.frame_files_path = folder_to_files_path(folder_path +"/", ".xml")
        self.parse()

    def parse_file(self, f):
        tree = ET.parse(f)
        root = tree.getroot()
        for child in root:
            if child.tag == "predicate":
                self.add_lemma(child)

    # add cannonical amr lemma to possible set of words including for aliases of the words
    def add_lemma(self, node):
        lemma = node.attrib["lemma"].replace("_", "-")
        self.frames.setdefault(lemma, [])
        #    self.frames[lemma] = set()
        for child in node:
            if child.tag == "roleset":
                sensed_predicate = child.attrib["id"]
                self.frames[lemma].append(sensed_predicate)
                true_lemma =  sensed_predicate.split(".")[0]
                if sensed_predicate not in self.frames.setdefault(true_lemma, []):
                    self.frames[true_lemma].append(sensed_predicate)
        if len(self.frames[lemma]) == 0:
            del self.frames[lemma]

    def get_frames(self):
        return self.frames

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

FIELDS_2009 = ["id", "form", "lemma", "plemma", "pos", "ppos", "feat", "pfeat", "head", "phead", "deprel", "pdeprel", "fillpred", "pred"]

import re

def parse_sentence(sentence_blob: str) -> Tuple[List[Dict[str, str]], List[Tuple[int, int]], List[str]]:
    """
    Parses a chunk of text in the SemEval SDP format.

    Each word in the sentence is returned as a dictionary with the following
    format:
    'id': '1',
    'form': 'Pierre',
    'lemma': 'Pierre',
    'pos': 'NNP',
    'head': '2',   # Note that this is the `syntactic` head.
    'deprel': 'nn',
    'top': '-',
    'pred': '+',
    'frame': 'named:x-c'

    Along with a list of arcs and their corresponding tags. Note that
    in semantic dependency parsing words can have more than one head
    (it is not a tree), meaning that the list of arcs and tags are
    not tied to the length of the sentence.
    """
    annotated_sentence = []
    arc_indices = []
    arc_tags = []
    predicates_indexes = []
    lines = [line.split("\t") for line in sentence_blob.split("\n")
             if line and not line.strip().startswith("#")]
    for line_idx, line in enumerate(lines):
        annotated_token = {k:v for k, v in zip(FIELDS_2009, line)}
        if annotated_token['fillpred'] == "Y":
            predicates_indexes.append(line_idx)
        annotated_sentence.append(annotated_token)

    for line_idx, line in enumerate(lines):
        for predicate_idx, arg in enumerate(line[len(FIELDS_2009):]):
            if arg != "_":
                arc_indices.append((line_idx, predicate_idx))
                arc_tags.append(arg)
    return annotated_sentence, arc_indices, arc_tags,predicates_indexes, sentence_blob


def lazy_parse(text: str):
    for sentence in text.split("\n\n"):
        if sentence:
            yield parse_sentence(sentence)


import json
import os

@DatasetReader.register("conll2009_en")
class Conll2009DatasetReader(DatasetReader):
    """
    Reads a file in the conllu Universal Dependencies format.

    Parameters
    ----------
    token_indexers : ``Dict[str, TokenIndexer]``, optional (default=``{"tokens": SingleIdTokenIndexer()}``)
        The token indexers to be applied to the words TextField.
    use_language_specific_pos : ``bool``, optional (default = False)
        Whether to use UD POS tags, or to use the language specific POS tags
        provided in the conllu format.
    """
    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 use_gold: bool = False,
                 filter:bool=True,
                 lazy: bool = False,
                 read_frame_new:bool=False,
                 data_folder = None) -> None:
        super().__init__(lazy)
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self.use_gold = use_gold
        self.filter = filter
        self.data_folder = data_folder
        self.lemma_to_sensed = self.read_frames(data_folder,read_frame_new)
        self.read_frame_new = read_frame_new
        print ("total number of lemma to senses:", len(self.lemma_to_sensed))
        self.annotated_sentences = []

    def save_frames(self,data_folder = None):
        if data_folder is None:
            data_folder = self.data_folder

        for lemma in self.lemma_to_sensed:
            assert len(self.lemma_to_sensed[lemma]) > 0,(lemma,self.lemma_to_sensed[lemma])

        with open(data_folder+'/senses.json', 'w+') as outfile:
            json.dump(self.lemma_to_sensed, outfile)

    def read_frames(self,data_folder,read_frame_new):
        if os.path.exists(data_folder+'/senses.json') and not read_frame_new:
            with open(data_folder+'/senses.json') as infile:

                print ("load saved senses dict")
                return defaultdict(lambda: [], **json.load(infile))

        print ("build senses dict")
        nbbank = PropbankReader(data_folder+"/nb_frames").get_frames()
        pbbank = PropbankReader(data_folder+"/pb_frames").get_frames()

        out = defaultdict(lambda:[],**nbbank)
        for lemma in pbbank:
            if lemma in out:
                for pred in pbbank[lemma]:
                    if pred not in out[lemma]:
                        out[lemma].append(pred)
            else:
                out[lemma] = pbbank[lemma]
        return  out

    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)
        training = "train" in file_path or "development" in file_path
        logger.info("Reading conll2009 srl data from: %s", file_path)
   #     print ("reading", file_path)
        with open(file_path, encoding="utf8", errors='ignore') as sdp_file:
            for annotated_sentence, directed_arc_indices, arc_tags , predicates_indexes, _ in lazy_parse(sdp_file.read()):
                # If there are no arc indices, skip this instance.
                if not directed_arc_indices and self.filter and "train" in file_path  :
                    continue
                self.annotated_sentences.append(annotated_sentence)
                tokens = [word["form"] for word in annotated_sentence]
                pred_candidates, sense_indexes, predicates = self.data_for_sense_prediction(annotated_sentence,training)
                pos_tags = [word["pos"] for word in annotated_sentence] if self.use_gold else [word["ppos"] for word in annotated_sentence]
                dep_tags = [word["deprel"] for word in annotated_sentence] if self.use_gold else [word["pdeprel"] for word in annotated_sentence]
                if "ood" not in file_path:
                    yield self.text_to_instance(tokens, predicates_indexes,pos_tags, dep_tags,directed_arc_indices, arc_tags,pred_candidates, sense_indexes, predicates)
                else:
                    yield self.text_to_instance(tokens, predicates_indexes,pos_tags, dep_tags,None, None,pred_candidates, None, None)



    def data_for_sense_prediction(self,annotated_sentence,training):
        pred_candidates = []
        predicates = [ word["pred"]  for word in annotated_sentence  if word["fillpred"] == "Y"]
        sense_indexes = []
        for word in annotated_sentence:
            if word["fillpred"] == "Y":
                pred = word["pred"]
                lemma = word["plemma"]
                if training and lemma in self.lemma_to_sensed and pred  in self.lemma_to_sensed[lemma] :
                    sense_indexes.append(self.lemma_to_sensed[lemma].index(pred ))
                    pred_candidates.append(self.lemma_to_sensed[lemma] )
                elif training and self.read_frame_new:
        #            print ("train adding",lemma,self.lemma_to_sensed[lemma] + [pred])
                    self.lemma_to_sensed[lemma].append(pred)
                    sense_indexes.append(self.lemma_to_sensed[lemma].index(pred ))
                    pred_candidates.append(self.lemma_to_sensed[lemma] )
                else:
                    if  lemma in self.lemma_to_sensed:
                        pred_candidates.append(self.lemma_to_sensed[lemma] )
                        sense_indexes.append(0)
                    else:
                 #       print ("test empty nothing similar",lemma,pred)
                        sense_indexes.append(0)
                        pred_candidates.append([lemma+".01"])

        return pred_candidates,sense_indexes,predicates
    @overrides
    def text_to_instance(self, # type: ignore
                         tokens: List[str],
                         predicates_indexes: List[int],
                         pos_tags: List[str] ,
                         dep_tags: List[str] ,
                         arc_indices: List[Tuple[int, int]] = None,
                         arc_tags: List[str] = None,
                         pred_candidates:List[List[str]] = None,
                         sense_indexes:List[int] = None,
                         predicates:List[str] = None,) -> Instance:
        # pylint: disable=arguments-differ
        fields: Dict[str, Field] = {}
        token_field = TextField([Token(t) for t in tokens], self._token_indexers)


        fields["tokens"] = token_field
        fields["predicate_indexes"] = MultiIndexField(predicates_indexes,label_namespace = "predicate_indexes",padding_value=-1)
        fields["metadata"] = MetadataField({"tokens": tokens})
        fields["pos_tags"] = SequenceLabelField(pos_tags, token_field, label_namespace="pos")
        fields["dep_tags"] = SequenceLabelField(dep_tags, token_field, label_namespace="dep")
        fields["predicate_candidates"] = MultiCandidatesSequence(pred_candidates, label_namespace="predicates")
        if arc_indices is not None and arc_tags is not None:
            fields["arc_tags"] = NonSquareAdjacencyField(arc_indices, token_field, fields["predicate_indexes"], arc_tags,label_namespace="arc_types")
            fields["predicates"] = IndexSequenceLabelField(predicates,label_namespace="predicates")
            fields["sense_indexes"] = MultiIndexField(sense_indexes, label_namespace="sense_indexes",padding_value=0)

        return Instance(fields)





def main():
    data_folder = "/afs/inf.ed.ac.uk/group/project/xchen13/datasets/2009_conll_p2/data/CoNLL2009-ST-English/"
    reader = Conll2009DatasetReader(data_folder = data_folder,read_frame_new=True)

    train_data = reader.read("/afs/inf.ed.ac.uk/group/project/xchen13/datasets/2009_conll_p2/data/CoNLL2009-ST-English/CoNLL2009-ST-English-train.txt")
    dev_data = reader.read("/afs/inf.ed.ac.uk/group/project/xchen13/datasets/2009_conll_p2/data/CoNLL2009-ST-English/CoNLL2009-ST-English-development.txt")

    reader.save_frames()

    reader = Conll2009DatasetReader(data_folder = data_folder)
    dev_data = reader.read("/afs/inf.ed.ac.uk/group/project/xchen13/datasets/2009_conll_p2/data/CoNLL2009-ST-English/CoNLL2009-ST-evaluation-English.txt")




if __name__ == "__main__":
    main()
