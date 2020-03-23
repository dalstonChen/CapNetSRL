from typing import Dict, Tuple, List
import logging,os

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
    predicates = []
    lines = [line.split("\t") for line in sentence_blob.split("\n")
             if line and not line.strip().startswith("#")]
    for line_idx, line in enumerate(lines):
        annotated_token = {k:v for k, v in zip(FIELDS_2009, line)}
        if annotated_token['fillpred'] == "Y":
            predicates_indexes.append(line_idx)
            predicates.append(annotated_token['pred'] )
            arc_indices.append([])
            arc_tags.append([])
        annotated_sentence.append(annotated_token)

    for line_idx, line in enumerate(lines):
        for predicate_idx, arg in enumerate(line[len(FIELDS_2009):]):
            if arg != "_":
                arc_indices[predicate_idx].append(line_idx)
                arc_tags[predicate_idx].append(arg)
    return sentence_blob, arc_indices, arc_tags,predicates


def lazy_parse(text: str):
    for sentence in text.split("\n\n"):
        if sentence:
            yield parse_sentence(sentence)


import json
import os

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
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self.annotated_sentences_sets = {}

    @overrides
    def _read(self, file_path: str):
        assert file_path not in self.annotated_sentences_sets
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)
        self.annotated_sentences_sets[file_path] = []
        training = "train" in file_path or "development" in file_path
        logger.info("Reading conll2009 srl data from: %s", file_path)
   #     print ("reading", file_path)
        with open(file_path) as sdp_file:
            return lazy_parse(sdp_file.read())

def anydup(thelist):
    seen = set()
    for x in thelist:
        if x in seen and x in ["A0","A1","A2","A3","A4","A5"]: return x
        seen.add(x)
    return None

def any_conti_vio(thelist):
    seen = set()
    for x in thelist:
        if "C-" in x and x[2:] not in seen: return x
        seen.add(x)
    return None

def any_ref_vio(thelist):
    seen = set()
    for x in thelist:
        seen.add(x)
    for x in thelist:
        if "R-" in x and x[2:] not in seen: return x
    return None
def check_constraints(data):
    unique_violated = {}
    continuation = {}
    ref_vio = {}
    for sentence_blob, arc_indices, arc_tags,predicates in data:
        for predicate ,arc_tags_per_predicates in zip(predicates,arc_tags):
            duplicates = anydup(arc_tags_per_predicates)
            if duplicates is not None:
                unique_violated.setdefault(duplicates, []).append((sentence_blob,predicate,arc_tags_per_predicates))

            continuation_violation = any_conti_vio(arc_tags_per_predicates)
            if continuation_violation is not None:
                continuation.setdefault(continuation_violation, []).append((sentence_blob,predicates,arc_tags_per_predicates))

            violation = any_ref_vio(arc_tags_per_predicates)
            if violation is not None:
                ref_vio.setdefault(violation, []).append((sentence_blob,predicates,arc_tags_per_predicates))

    unique_violated =[(k,unique_violated[k]) for k in  sorted(unique_violated, key=lambda k: len(unique_violated[k]),reverse=True)]
    continuation =[(k,continuation[k]) for k in  sorted(continuation, key=lambda k: len(continuation[k]),reverse=True)]
    ref_vio =[(k,ref_vio[k]) for k in  sorted(ref_vio, key=lambda k: len(ref_vio[k]),reverse=True)]
    return unique_violated,continuation,ref_vio


def check_file(file,data):
    with open(file+".unique_viol","w+") as u_file:
        with open(file+".continu_viol","w+") as c_file:
            with open(file+".ref_viol","w+") as r_file:
                unique_violated, continuation,ref_vio = check_constraints(data)
                u_file.write("total violations\t"+str(sum ([len(violations) for arg,violations in unique_violated]) )+"\n")
                for arg, violations in unique_violated:
                    u_file.write(arg+"\t"+str(len(violations))+"\n")
                    for sentence_blob,predicates,arc_tags_per_predicates in violations:
                        u_file.write(str(predicates))
                        u_file.write("\n")
                        u_file.write(str(arc_tags_per_predicates))
                        u_file.write("\n")
                        u_file.write(sentence_blob)
                        u_file.write("\n")

                c_file.write("total violations\t"+str(sum([len(violations) for arg,violations in continuation] ))+"\n")
                for arg , violations in continuation:
                    c_file.write(arg+"\t"+str(len(violations))+"\n")
                    for sentence_blob,predicates,arc_tags_per_predicates in violations:
                        c_file.write(str(predicates))
                        c_file.write("\n")
                        c_file.write(str(arc_tags_per_predicates))
                        c_file.write("\n")
                        c_file.write(sentence_blob)
                        c_file.write("\n")

                r_file.write("total violations\t" + str(
                    sum([len(violations) for arg, violations in ref_vio])) + "\n")
                for arg, violations in ref_vio:
                    r_file.write(arg + "\t" + str(len(violations)) + "\n")
                    for sentence_blob, predicates, arc_tags_per_predicates in violations:
                        r_file.write(str(predicates))
                        r_file.write("\n")
                        r_file.write(str(arc_tags_per_predicates))
                        r_file.write("\n")
                        r_file.write(sentence_blob)
                        r_file.write("\n")

                print("printing "+file.split("/")[-1]+" set constrain violations")
                print("unique role violations frequency")
                for arg, violations in unique_violated:
                    print(arg, len(violations))
                print("continuation role violations frequency")
                for arg, violations in continuation:
                    print(arg, len(violations))

                return unique_violated, continuation
def main():
    data_folder = "/afs/inf.ed.ac.uk/user/s15/s1544871/Data/2009_conll_p2/data/english_dev_best/"
    gold_file = data_folder+"CoNLL2009-ST-English-development.txt"
    base_file = data_folder+"CoNLL2009-ST-English-development.predict0"
    refined_file = data_folder+"CoNLL2009-ST-English-development.predict2"

    def arg_errors():
        errors = {}  # arg -> base_arg -> int
        all_errors = {}  # arg -> base_arg -> int
        error_fixed = {}  # arg -> base_arg -> examples
        error_fixed_entirely = {}  # arg -> base_arg -> examples
        error_not_fixed = {}  # arg -> base_arg -> examples

        fixed_freq = {}
        unfixed_freq = {}
        num_errors = {}

        for base_instance, refine_instance, gold_instance in zip(base_data, refined_data, gold_data):
            sentence_blob, arc_indices_base, arc_tags_base, predicates = base_instance
            sentence_blob, arc_indices_refine, arc_tags_refine, predicates = refine_instance
            sentence_blob, arc_indices_gold, arc_tags_gold, predicates = gold_instance

            snt = " ".join([line.split("\t")[1] for line in sentence_blob.split("\n")])
            sentence_blob = snt + "\n" + sentence_blob
            seq_len = len(sentence_blob.split("\n"))
            for i, predicate in enumerate(predicates):
                arcs_base = ["_"] * seq_len
                for index, tag in zip(arc_indices_base[i], arc_tags_base[i]):
                    arcs_base[index] = tag
                arcs_base = tuple(arcs_base)

                arcs_refine = ["_"] * seq_len
                for index, tag in zip(arc_indices_refine[i], arc_tags_refine[i]):
                    arcs_refine[index] = tag
                arcs_refine = tuple(arcs_refine)

                arcs_gold = ["_"] * seq_len
                for index, tag in zip(arc_indices_gold[i], arc_tags_gold[i]):
                    arcs_gold[index] = tag
                arcs_gold = tuple(arcs_gold)

                for j in range(seq_len):
                    if arcs_base[j] != arcs_gold[j]:
                        errors[(arcs_gold[j], arcs_base[j])] = errors.setdefault((arcs_gold[j], arcs_base[j]), 0) + 1
                        all_errors.setdefault((arcs_gold[j], arcs_base[j]), []).append(
                            (sentence_blob, predicate, arcs_base, arcs_refine, arcs_gold))
                        if arcs_gold[j] == arcs_refine[j]:
                            error_fixed.setdefault((arcs_gold[j], arcs_base[j]), []).append(
                                (sentence_blob, predicate, arcs_base, arcs_refine, arcs_gold))

                            if arcs_gold == arcs_refine:
                                error_fixed_entirely.setdefault((arcs_gold[j], arcs_base[j]), []).append(
                                    (sentence_blob, predicate, arcs_base, arcs_refine, arcs_gold))
                        else:
                            error_not_fixed.setdefault((arcs_gold[j], arcs_base[j]), []).append(
                                (sentence_blob, predicate, arcs_base, arcs_refine, arcs_gold))

            #    for arc in arcs_base:

        with open(data_folder + "all_errors", "w+") as file:
            sorted_errors = [(k, all_errors[k]) for k in
                             sorted(all_errors, key=lambda k: len(all_errors[k]), reverse=True)]
            file.write("total error: " + str(sum([len(examples) for gold_base, examples in sorted_errors])) + "\n")
            file.write("Formats are\npredicate\narcs_base\narcs_refine\narcs_gold\nsentence\n\n")
            for gold_base, examples in sorted_errors:
                file.write(str(gold_base) + "\t" + str(len(examples)) + "\n")
                num_errors[gold_base] = len(examples)
                for sentence_blob, predicate, arcs_base, arcs_refine, arcs_gold in examples:
                    file.write(str(predicate))
                    file.write("\n")
                    file.write(str(arcs_base))
                    file.write("\n")
                    file.write(str(arcs_refine))
                    file.write("\n")
                    file.write(str(arcs_gold))
                    file.write("\n")
                    file.write(sentence_blob)
                    file.write("\n")

        with open(data_folder + "fixed", "w+") as file:
            sorted_errors = [(k, error_fixed[k]) for k in
                             sorted(error_fixed, key=lambda k: len(error_fixed[k]), reverse=True)]
            file.write("total error: " + str(sum([len(examples) for gold_base, examples in sorted_errors])) + "\n")
            file.write("Formats are\npredicate\narcs_base\narcs_refine\narcs_gold\nsentence\n\n")
            for gold_base, examples in sorted_errors:
                file.write(str(gold_base) + "\t" + str(len(examples)) + "\n")
                fixed_freq[gold_base] = 1.0 * len(examples) / num_errors[gold_base]
                for sentence_blob, predicate, arcs_base, arcs_refine, arcs_gold in examples:
                    file.write(str(predicate))
                    file.write("\n")
                    file.write(str(arcs_base))
                    file.write("\n")
                    file.write(str(arcs_refine))
                    file.write("\n")
                    file.write(str(arcs_gold))
                    file.write("\n")
                    file.write(sentence_blob)
                    file.write("\n")

        with open(data_folder + "unfixed", "w+") as file:
            sorted_errors = [(k, error_not_fixed[k]) for k in
                             sorted(error_not_fixed, key=lambda k: len(error_not_fixed[k]), reverse=True)]
            file.write("total error: " + str(sum([len(examples) for gold_base, examples in sorted_errors])) + "\n")
            file.write("Formats are\npredicate\narcs_base\narcs_refine\narcs_gold\nsentence\n\n")
            for gold_base, examples in sorted_errors:
                file.write(str(gold_base) + "\t" + str(len(examples)) + "\n")
                unfixed_freq[gold_base] = 1.0 * len(examples) / num_errors[gold_base]
                for sentence_blob, predicate, arcs_base, arcs_refine, arcs_gold in examples:
                    file.write(str(predicate))
                    file.write("\n")
                    file.write(str(arcs_base))
                    file.write("\n")
                    file.write(str(arcs_refine))
                    file.write("\n")
                    file.write(str(arcs_gold))
                    file.write("\n")
                    file.write(sentence_blob)
                    file.write("\n")

        with open(data_folder + "complete_fixed", "w+") as file:
            sorted_errors = [(k, error_fixed_entirely[k]) for k in
                             sorted(error_fixed_entirely, key=lambda k: len(error_fixed_entirely[k]), reverse=True)]
            file.write("total error: " + str(sum([len(examples) for gold_base, examples in sorted_errors])) + "\n")
            file.write("Formats are\npredicate\narcs_base\narcs_refine\narcs_gold\nsentence\n\n")
            for gold_base, examples in sorted_errors:
                file.write(str(gold_base) + "\t" + str(len(examples)) + "\n")
                for sentence_blob, predicate, arcs_base, arcs_refine, arcs_gold in examples:
                    file.write(str(predicate))
                    file.write("\n")
                    file.write(str(arcs_base))
                    file.write("\n")
                    file.write(str(arcs_refine))
                    file.write("\n")
                    file.write(str(arcs_gold))
                    file.write("\n")
                    file.write(sentence_blob)
                    file.write("\n")

        print("fixed_freq", list(sorted(fixed_freq.items(), key=lambda kv: kv[1], reverse=True)))
        print("unfixed_freq", list(sorted(unfixed_freq.items(), key=lambda kv: kv[1], reverse=True)))
    
    reader = Conll2009DatasetReader()
    gold_data = reader.read(gold_file)
    unique_violated, continuation = check_file(gold_file,gold_data)



    base_data = reader.read(base_file)
    unique_violated, continuation = check_file(base_file,base_data)
    refined_data = reader.read(refined_file)
    unique_violated, continuation = check_file(refined_file,refined_data)

    arg_errors()

    errors = {}  # sense -> base_sense -> int
    all_errors = {}  # sense -> base_sense -> int
    error_fixed = {}  # sense -> base_sense -> examples
    error_fixed_entirely = {}  # sense -> base_ -> examples
    error_not_fixed = {}  # sense -> base_sense -> examples

    fixed_freq = {}
    unfixed_freq = {}
    num_errors = {}

    for base_instance, refine_instance, gold_instance in zip(base_data, refined_data, gold_data):
        sentence_blob, arc_indices_base, arc_tags_base, predicates_base = base_instance
        sentence_blob, arc_indices_refine, arc_tags_refine, predicates_refine = refine_instance
        sentence_blob, arc_indices_gold, arc_tags_gold, predicates_gold = gold_instance

        snt = " ".join([line.split("\t")[1] for line in sentence_blob.split("\n")])
        sentence_blob = snt + "\n" + sentence_blob
        seq_len = len(sentence_blob.split("\n"))
        for i, (predicate_base,predicate_refine,predicate_gold) in enumerate(zip(predicates_base,predicates_refine,predicates_gold)):

            arcs_base = ["_"] * seq_len
            for index, tag in zip(arc_indices_base[i], arc_tags_base[i]):
                arcs_base[index] = tag
            arcs_base = tuple(arcs_base)

            arcs_refine = ["_"] * seq_len
            for index, tag in zip(arc_indices_refine[i], arc_tags_refine[i]):
                arcs_refine[index] = tag
            arcs_refine = tuple(arcs_refine)

            arcs_gold = ["_"] * seq_len
            for index, tag in zip(arc_indices_gold[i], arc_tags_gold[i]):
                arcs_gold[index] = tag
            arcs_gold = tuple(arcs_gold)

            if predicate_base != predicate_gold:
                errors[(predicate_base, predicate_gold)] = errors.setdefault((predicate_base, predicate_gold), 0) + 1
                all_errors.setdefault((predicate_base, predicate_gold), []).append(
                    (sentence_blob, predicate_base, predicate_refine,predicate_gold,arcs_base, arcs_refine, arcs_gold))
                if predicate_gold ==predicate_refine:
                    error_fixed.setdefault((predicate_base, predicate_gold), []).append(
                    (sentence_blob, predicate_base, predicate_refine,predicate_gold,arcs_base, arcs_refine, arcs_gold))

                    if arcs_gold == arcs_refine:
                        error_fixed_entirely.setdefault((predicate_base, predicate_gold), []).append(
                    (sentence_blob, predicate_base, predicate_refine,predicate_gold,arcs_base, arcs_refine, arcs_gold))
                else:
                    error_not_fixed.setdefault((predicate_base, predicate_gold), []).append(
                    (sentence_blob, predicate_base, predicate_refine,predicate_gold,arcs_base, arcs_refine, arcs_gold))

        #    for arc in arcs_base:

    with open(data_folder + "all_errors_pred", "w+") as file:
        sorted_errors = [(k, all_errors[k]) for k in
                         sorted(all_errors, key=lambda k: len(all_errors[k]), reverse=True)]
        file.write("total error: " + str(sum([len(examples) for gold_base, examples in sorted_errors])) + "\n")
        file.write("Formats are\npredicate\narcs_base\narcs_refine\narcs_gold\nsentence\n\n")
        for gold_base, examples in sorted_errors:
            file.write(str(gold_base) + "\t" + str(len(examples)) + "\n")
            num_errors[gold_base] = len(examples)
            for sentence_blob, predicate_base, predicate_refine,predicate_gold, arcs_base, arcs_refine, arcs_gold in examples:
                file.write(str(predicate_base))
                file.write("\n")
                file.write(str(arcs_base))
                file.write("\n")
                file.write(str(predicate_refine))
                file.write("\n")
                file.write(str(arcs_refine))
                file.write("\n")
                file.write(str(predicate_gold))
                file.write("\n")
                file.write(str(arcs_gold))
                file.write("\n")
                file.write(sentence_blob)
                file.write("\n")

    with open(data_folder + "error_fixed_pred", "w+") as file:
        sorted_errors = [(k, error_fixed[k]) for k in
                         sorted(error_fixed, key=lambda k: len(error_fixed[k]), reverse=True)]
        file.write(
            "total error: " + str(sum([len(examples) for gold_base, examples in sorted_errors])) + "\n")
        file.write("Formats are\npredicate\narcs_base\narcs_refine\narcs_gold\nsentence\n\n")
        for gold_base, examples in sorted_errors:
            file.write(str(gold_base) + "\t" + str(len(examples)) + "\n")
            fixed_freq[gold_base] = 1.0 * len(examples) / num_errors[gold_base]
            for sentence_blob, predicate_base, predicate_refine, predicate_gold, arcs_base, arcs_refine, arcs_gold in examples:
                file.write(str(predicate_base))
                file.write("\n")
                file.write(str(arcs_base))
                file.write("\n")
                file.write(str(predicate_refine))
                file.write("\n")
                file.write(str(arcs_refine))
                file.write("\n")
                file.write(str(predicate_gold))
                file.write("\n")
                file.write(str(arcs_gold))
                file.write("\n")
                file.write(sentence_blob)
                file.write("\n")
    with open(data_folder + "error_fixed_entirely_pred", "w+") as file:
        sorted_errors = [(k, error_fixed_entirely[k]) for k in
                         sorted(error_fixed_entirely, key=lambda k: len(error_fixed_entirely[k]), reverse=True)]
        file.write(
            "total error: " + str(sum([len(examples) for gold_base, examples in sorted_errors])) + "\n")
        file.write("Formats are\npredicate\narcs_base\narcs_refine\narcs_gold\nsentence\n\n")
        for gold_base, examples in sorted_errors:
            file.write(str(gold_base) + "\t" + str(len(examples)) + "\n")
            for sentence_blob, predicate_base, predicate_refine, predicate_gold, arcs_base, arcs_refine, arcs_gold in examples:
                file.write(str(predicate_base))
                file.write("\n")
                file.write(str(arcs_base))
                file.write("\n")
                file.write(str(predicate_refine))
                file.write("\n")
                file.write(str(arcs_refine))
                file.write("\n")
                file.write(str(predicate_gold))
                file.write("\n")
                file.write(str(arcs_gold))
                file.write("\n")
                file.write(sentence_blob)
                file.write("\n")

    with open(data_folder + "error_not_fixed_pred", "w+") as file:
        sorted_errors = [(k, error_not_fixed[k]) for k in
                         sorted(error_not_fixed, key=lambda k: len(error_not_fixed[k]), reverse=True)]
        file.write(
            "total error: " + str(sum([len(examples) for gold_base, examples in sorted_errors])) + "\n")
        file.write("Formats are\npredicate\narcs_base\narcs_refine\narcs_gold\nsentence\n\n")
        for gold_base, examples in sorted_errors:
            file.write(str(gold_base) + "\t" + str(len(examples)) + "\n")
            unfixed_freq[gold_base] = 1.0 * len(examples) / num_errors[gold_base]
            for sentence_blob, predicate_base, predicate_refine, predicate_gold, arcs_base, arcs_refine, arcs_gold in examples:
                file.write(str(predicate_base))
                file.write("\n")
                file.write(str(arcs_base))
                file.write("\n")
                file.write(str(predicate_refine))
                file.write("\n")
                file.write(str(arcs_refine))
                file.write("\n")
                file.write(str(predicate_gold))
                file.write("\n")
                file.write(str(arcs_gold))
                file.write("\n")
                file.write(sentence_blob)
                file.write("\n")

    print("fixed_freq", list(sorted(fixed_freq.items(), key=lambda kv: kv[1], reverse=True)))
    print("unfixed_freq", list(sorted(unfixed_freq.items(), key=lambda kv: kv[1], reverse=True)))
if __name__ == "__main__":
    main()

    def do():
        n_A2 = 0
        n_A1 = 0
        with open("/afs/inf.ed.ac.uk/user/s15/s1544871/Data/2009_conll_p2/data/CoNLL2009-ST-English/CoNLL2009-ST-English-train.txt") as file:
            line = file.readline()
            while line != "":
                if line and not line.strip().startswith("#"):
                    spliited = line.split("\t")
                    if len(spliited)>1  and "IN" == spliited[5]: #"ADV" == spliited[10] : # and "of" == spliited[1] "IN" == spliited[5] and
                        if "\tA2" in line:
                            n_A2  = n_A2 +1
                        if "\tA1" in line:
                            n_A1  = n_A1 +1
                line = file.readline()
        print ("ADV n_A1,n_A2",n_A1,n_A2)