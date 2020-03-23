import sys
sys.path.append("/afs/inf.ed.ac.uk/user/x/xchen13/project/SRL/")

from myallennlp.dataset_readers.conll2009 import lazy_parse
from allennlp.common.file_utils import cached_path

gold_path = '/afs/inf.ed.ac.uk/group/project/xchen13/datasets/2009_conll_p2/data/CoNLL2009-ST-English/CoNLL2009-ST-English-development.txt'
#capsule_path = '/disk/scratch1/xchen13/single_iter2_weight_conll09_600ep/conll09_test.predict'
#baseline_path = '/disk/scratch1/xchen13/single_ave_conll09_600ep/conll09_test.predict'
baseline_path = '/disk/scratch1/xchen13/single_iter2_weight_conll09_global_600ep/conll09_dev_seventh_iter.predict'
capsule_path = '/disk/scratch1/xchen13/single_iter2_weight_conll09_global_600ep/conll09_dev_eighth_iter.predict'
gold_path = cached_path(gold_path)
capsule_path = cached_path(capsule_path)
baseline_path = cached_path(baseline_path)

def count_dup_arg(directed_arc_indices, arc_tags):
    predicates = {}
    cnt = 0
    multi_ans = []
    for arc_indice, arc_tag in zip(directed_arc_indices, arc_tags):
        if predicates.get(arc_indice[1]) == None:
            predicates[arc_indice[1]] = {}
        if predicates[arc_indice[1]].get(arc_tag) == None:
            predicates[arc_indice[1]][arc_tag] = 0
        predicates[arc_indice[1]][arc_tag] += 1
        if predicates[arc_indice[1]][arc_tag] == 2:
            #if arc_tag not in ['A0','A1','A2','A3','A4','A5']: continue
            #print(arc_tag)
            cnt += 1
            print(predicates[arc_indice[1]],arc_indice[1])
            multi_ans.append((arc_indice[0], arc_tag))
    return cnt, multi_ans

def Conll_print(annotated_sentence):
    for line in annotated_sentence:
        lll = []
        for key, value in line.items():
            lll.append(value)
        print("\t".join(lll))

with open(gold_path, encoding="utf8", errors='ignore') as gold_file, \
    open(baseline_path, encoding="utf8", errors='ignore') as baseline_file, \
    open(capsule_path, encoding="utf8", errors='ignore') as capsule_file:
    #annotated_sentence, directed_arc_indices, arc_tags , predicates_indexes, sentence_blob
    cnt_gold = cnt_baseline = cnt_capsule = 0
    for gold, baseline, capsule in zip(lazy_parse(gold_file.read()), lazy_parse(baseline_file.read()), lazy_parse(capsule_file.read())):
        #Conll_print(gold[0])
        #print(gold[1])
        #print(gold[2])
        #print('gold')
        cnt, gold_pred = count_dup_arg(gold[1], gold[2])
        cnt_gold += cnt
        #print('baseline')
        cnt, baseline_pred= count_dup_arg(baseline[1], baseline[2])
        #if baseline_pred != gold_pred:
        cnt_baseline += cnt
        #print('capsule')
        cnt, capsule_pred = count_dup_arg(capsule[1], capsule[2])
        #if capsule_pred != gold_pred:
        cnt_capsule += cnt
        #print(gold_pred == capsule_pred)
        #print()
    print ("cnt_gold, cnt_baseline, cnt_capsule:", cnt_gold, cnt_baseline, cnt_capsule)