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
    return annotated_sentence, arc_indices, arc_tags,predicates_indexes


def lazy_parse(text: str):
    for sentence in text.split("\n\n"):
        if sentence:
            yield parse_sentence(sentence)


import json
import os

from myallennlp.dataset_readers.conll2009 import Conll2009DatasetReader
@DatasetReader.register("conll2009_de")
class Conll2009DeDatasetReader(Conll2009DatasetReader):
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

    def save_frames(self,data_folder = None):
        if data_folder is None:
            data_folder = self.data_folder

        for lemma in self.lemma_to_sensed:
            assert len(self.lemma_to_sensed[lemma]) > 0,(lemma,self.lemma_to_sensed[lemma])

        print ("total number of lemma to senses to save:", len(self.lemma_to_sensed))
        with open(data_folder+'/senses.json', 'w+') as outfile:
            json.dump(self.lemma_to_sensed, outfile)

    def read_frames(self,data_folder,read_frame_new):
        if os.path.exists(data_folder+'/senses.json') and not read_frame_new:
            with open(data_folder+'/senses.json') as infile:

                print ("load saved senses dict")
                return defaultdict(lambda: [], **json.load(infile))

        print ("build senses dict")
        esbank = PropbankReader(data_folder+"/frames").get_frames()

        out = defaultdict(lambda:[],**esbank)
        return  out

    @overrides
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
                elif training and self.read_frame_new :
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
                        pred_candidates.append([lemma+".1"])


        return pred_candidates,sense_indexes,predicates




def main():
    data_folder = "/afs/inf.ed.ac.uk/group/project/xchen13/datasets/2009_conll_p1/data/CoNLL2009-ST-German/"
    reader = Conll2009DeDatasetReader(data_folder = data_folder,read_frame_new=True)

    train_data = reader.read(data_folder+"CoNLL2009-ST-German-train.txt")
    dev_data = reader.read(data_folder+"CoNLL2009-ST-German-development.txt")

    reader.save_frames()

    reader = Conll2009DeDatasetReader(data_folder = data_folder)

    dev_data = reader.read(data_folder +"CoNLL2009-ST-evaluation-German.txt")



if __name__ == "__main__":
    main()