from typing import List, Dict

from overrides import overrides

from allennlp.common.util import JsonDict, sanitize, group_by_count
from allennlp.data import DatasetReader, Instance
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter
from allennlp.models import Model
from allennlp.predictors.predictor import Predictor

import json
from allennlp.models.archival import Archive, load_archive
from allennlp.common.util import import_submodules
FIELDS_2009 = ["id", "form", "lemma", "plemma", "pos", "ppos", "feat", "pfeat", "head", "phead", "deprel", "pdeprel", "fillpred"]#, "pred"]

@Predictor.register("dependency_srl")
class Conll2009_Predictor(Predictor): #CoNLL2009-ST-evaluation-English  CoNLL2009-ST-English-development
    def __init__(self, model: Model, dataset_reader: DatasetReader,
                 output_file_path= "/disk/scratch1/xchen13/es_advanced_iter2_argcaplinear_global_conll09_100ep/conll09_dev.predict") -> None:
        super().__init__(model, dataset_reader)
        #self.result = []
        self.crt_instance_id_start = 0
        #print(self._dataset_reader.type)

        if output_file_path is not None:
            self.set_files(output_file_path)

    def set_files(self,output_file_path):

        self.conll_format_file_path = output_file_path

        self.file_out_g = open(self.conll_format_file_path +"_g", 'w+', encoding="utf8", errors='ignore')
        self.file_out_t= open(self.conll_format_file_path, 'w+', encoding="utf8", errors='ignore')
        self.file_out = []


    def to_conll_format(self, instance, annotated_sentence, output):
        #    print("output_key", output.keys())

        sentence_len = len(annotated_sentence)

        EXCEPT_LIST = ['@@PADDING@@', '@@UNKNOWN@@', '_']
        pred_candidates = instance["predicate_candidates"].labels
        predicate_indexes = instance["predicate_indexes"].labels

        max_num_slots = len(pred_candidates)
        predicates = ["_"] * sentence_len

        if "sense_argmax_g" in output:
            sense_argmax = output["sense_argmax_g"]
            predicted_arc_tags = output["predicted_arc_tags_g"]

            for sense_idx, idx, pred_candidate in zip(sense_argmax, predicate_indexes, pred_candidates):
                if pred_candidate[sense_idx] not in EXCEPT_LIST:
                    predicates[idx] = pred_candidate[sense_idx]
            for idx in range(sentence_len):
                word = annotated_sentence[idx]
                arg_slots = ['_'] * max_num_slots
                for y in range(max_num_slots):  # output["arc_tags"].shape[1]):
                    if predicted_arc_tags[idx][y] != -1:
                        arg_slots[y] = self._model.vocab.get_index_to_token_vocabulary("arc_types")[
                            predicted_arc_tags[idx][y]]

                pred_label = predicates[idx]
                string = '\t'.join([word[type] for type in FIELDS_2009] + [pred_label] + arg_slots)
                self.file_out_g.write(string + '\n')
            self.file_out_g.write('\n')

        if "sense_argmax" in output:
            sense_argmax = output["sense_argmax"]
            predicted_arc_tags = output["predicted_arc_tags"]

            for sense_idx, idx, pred_candidate in zip(sense_argmax, predicate_indexes, pred_candidates):
                if sense_idx > len(pred_candidate) - 1:
                    print(sense_idx,pred_candidate)
                    sense_idx = len(pred_candidate) - 1
                if pred_candidate[sense_idx] not in EXCEPT_LIST:
                    predicates[idx] = pred_candidate[sense_idx]
            for idx in range(sentence_len):
                word = annotated_sentence[idx]
                arg_slots = ['_'] * max_num_slots
                for y in range(max_num_slots):  # output["arc_tags"].shape[1]):
                    if predicted_arc_tags[idx][y] != -1:
                        arg_slots[y] = self._model.vocab.get_index_to_token_vocabulary("arc_types")[
                            predicted_arc_tags[idx][y]]

                pred_label = predicates[idx]
                string = '\t'.join([word[type] for type in FIELDS_2009] + [pred_label] + arg_slots)
                self.file_out_t.write(string + '\n')
            self.file_out_t.write('\n')

        for i in range(10):
            if "sense_argmax"+str(i) in output:
                if i >= len(self.file_out):
                    self.file_out.append(open(self.conll_format_file_path+str(i), 'w+', encoding="utf8", errors='ignore'))
                sense_argmax = output["sense_argmax"+str(i) ]
                predicted_arc_tags = output["predicted_arc_tags"+str(i)]

                for sense_idx,idx,pred_candidate in zip(sense_argmax,predicate_indexes,pred_candidates):
                    if pred_candidate[sense_idx] not in EXCEPT_LIST:
                        predicates[idx] = pred_candidate[sense_idx]
                for idx in range(sentence_len):
                    word = annotated_sentence[idx]
                    arg_slots = ['_'] * max_num_slots
                    for y in range(max_num_slots):  # output["arc_tags"].shape[1]):
                        if predicted_arc_tags[idx][y] != -1:
                            arg_slots[y] = self._model.vocab.get_index_to_token_vocabulary("arc_types")[predicted_arc_tags[idx][y]]

                    pred_label = predicates[idx]
                    string = '\t'.join([word[type] for type in FIELDS_2009] + [pred_label] + arg_slots)
                    self.file_out[i].write(string + '\n')
                self.file_out[i].write('\n')
        return

    @overrides
    def predict_instance(self, instances: Instance) -> JsonDict:

        return self.predict_batch_instance([instances])[0]

    @overrides
    def predict_batch_instance(self, instances: List[Instance]) -> List[JsonDict]:
       # print ("last_instances", instances[-1])
       # print ("last_", self._dataset_reader.annotated_sentences[-1])
        #print('*****')
        #print(len(instances))
        #print(instances)
        outputs = self._model.forward_on_instances(instances)
        outputs = sanitize(outputs)
        for instance,annotated_sentence, output in zip(instances,self._dataset_reader.annotated_sentences[self.crt_instance_id_start:self.crt_instance_id_start+len(outputs)], outputs):
            #print(self.crt_instance_id_start, len(outputs))
            #print("output", output["tokens"])
            #assert False
            self.to_conll_format(instance,annotated_sentence, output)
        self.crt_instance_id_start += len(outputs)
        #self.result += outputs
        return outputs
