import sys
sys.path.append("/afs/inf.ed.ac.uk/user/x/xchen13/project/SRL/")

from myallennlp.dataset_readers.conll2009 import lazy_parse
from allennlp.common.file_utils import cached_path
import numpy as np

gold_path = '/afs/inf.ed.ac.uk/group/project/xchen13/datasets/2009_conll_p2/data/CoNLL2009-ST-English/CoNLL2009-ST-English-development.txt'
#iter1_path = '/disk/scratch1/xchen13/single_iter2_weight_conll09_600ep/conll09_dev_first_iter.predict'
#iter2_path = '/disk/scratch1/xchen13/single_iter2_weight_conll09_600ep/conll09_dev.predict'
#iter1_path = '/disk/scratch1/xchen13/single_iter2_weight_conll09_600ep/conll09_dev.predict'
#iter2_path = '/disk/scratch1/xchen13/single_iter2_weight_conll09_600ep/conll09_dev_third_iter.predict'
iter1_path = '/disk/scratch1/xchen13/single_iter2_weight_conll09_600ep/conll09_dev_first_iter.predict'
iter2_path = '/disk/scratch1/xchen13/single_iter2_weight_conll09_600ep/conll09_dev.predict'
gold_path = cached_path(gold_path)
iter1_path = cached_path(iter1_path)
iter2_path = cached_path(iter2_path)

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
        if predicates[arc_indice[1]][arc_tag] > 1:
            cnt += 1
            print(predicates[arc_indice[1]],arc_indice[1])
            multi_ans.append((arc_indice[0], arc_tag))
    return cnt, multi_ans
def add_one(role1, role2, value, iter1toiter2_role_cnt):
    if iter1toiter2_role_cnt.get(role1) == None:
        iter1toiter2_role_cnt[role1] = {}
    if iter1toiter2_role_cnt[role1].get(role2) == None:
        iter1toiter2_role_cnt[role1][role2] = 0
    iter1toiter2_role_cnt[role1][role2] += value



def dict2table(d, bar = None):
    roles = []
    for key,value in d.items():
        roles.append(key)
        for k,v in value.items():
            roles.append(k)
    roles = list(set(roles))
    roles.sort()
    roles.remove('None')
    roles.append('None')
    if bar!=None:
        remove_roles = []
        for r in roles:
            s = 0
            if d.get(r) != None:
                s = np.sum(np.abs([int(v) for k,v in d[r].items()]))
            for role in roles:
                if d.get(role) != None and d[role].get(r) != None:
                    s += np.abs(int(d[role][r]))
            if s < bar:
                remove_roles.append(r)
            print(r,s)
        for rr in remove_roles:
            roles.remove(rr)

    print (roles)
    trans_matrix = []
    print("\t".join(['Roles'] + roles))
    for role in roles:
        print_list = [role]
        if d.get(role) == None:
            print_list += ['0'] * len(roles)
        else:
            ddd = d[role]
            for r in roles:
                if ddd.get(r) == None:
                    print_list.append('0')
                else:
                    print_list.append(str(ddd[r]))
        print("\t".join(print_list))
        trans_matrix.append(print_list[1:])
    print(trans_matrix)

def Conll_print(sentence_blob, pred_idx, pred_position):
    lines = [line.split("\t") for line in sentence_blob.split("\n")
             if line and not line.strip().startswith("#")]
    res = []
    for line_idx, line in enumerate(lines):
        res_line = line[:12]
        if line_idx == pred_position:
            res_line.append('Y')
            res_line.append(line[13])
        else:
            res_line.append('_')
            res_line.append('_')
        res_line.append(line[14+pred_idx])
        res.append("\t".join(res_line))
    print("\n".join(res) + '\n\n')

with open(gold_path, encoding="utf8", errors='ignore') as gold_file, \
    open(iter1_path, encoding="utf8", errors='ignore') as iter1_file, \
    open(iter2_path, encoding="utf8", errors='ignore') as iter2_file:
    #annotated_sentence, directed_arc_indices, arc_tags , predicates_indexes, sentence_blob
    iter1toiter2_role_cnt = {}
    improve_role_cnt = {}
    for gold, iter1, iter2 in zip(lazy_parse(gold_file.read()), lazy_parse(iter1_file.read()), lazy_parse(iter2_file.read())):
        for line_idx in range(len(gold[0])):
            for predicate_idx in range(len(gold[3])):
                role1 = 'None'
                if (line_idx, predicate_idx) in iter1[1]:
                    idx = iter1[1].index((line_idx, predicate_idx))
                    role1 = iter1[2][idx]
                role2 = 'None'
                if (line_idx, predicate_idx) in iter2[1]:
                    idx = iter2[1].index((line_idx, predicate_idx))
                    role2 = iter2[2][idx]
                add_one(role1, role2, 1, iter1toiter2_role_cnt)
                gold_role = 'None'
                if (line_idx, predicate_idx) in gold[1]:
                    idx = gold[1].index((line_idx, predicate_idx))
                    gold_role = gold[2][idx]
                #if role1 != role2:
                    #if role1 == gold_role:
                    #    add_one(role1, role2, -1, improve_role_cnt)
                if role2 == gold_role:
                    add_one(role1, role2, 1, improve_role_cnt)
                else:
                    add_one(role1, role2, -1, improve_role_cnt)

                if role1!='None' and role2!='None' and role1 != role2 and role2 == gold_role and len(gold[0])<35 and gold[0][gold[3][predicate_idx]]['pos'][0]=='V':
                    ccc = 0
                    for (iter1_idx, iter1_pred_idx), iter1_tag in zip(iter1[1], iter1[2]):
                        if iter1_pred_idx == predicate_idx:
                            if iter1_tag == role2:
                                ccc += 1
                    bbb = 0
                    for (gold_idx, gold_pred_idx), gold_tag in zip(gold[1], gold[2]):
                        if gold_pred_idx == predicate_idx:
                            if gold_tag == role2:
                                bbb += 1
                    #if ccc == 1:# and bbb == 1:
                    print(line_idx+1, gold[3][predicate_idx]+1, gold_role, role1, role2)
                    print(gold[4])
                    Conll_print(gold[4], predicate_idx, gold[3][predicate_idx])
                    Conll_print(iter1[4], predicate_idx, iter1[3][predicate_idx])
                    Conll_print(iter2[4], predicate_idx, iter2[3][predicate_idx])




    print('iter1toiter2_role_cnt')
    dict2table(iter1toiter2_role_cnt,bar = 50)
    print()
    print('improve_role_cnt')
    dict2table(improve_role_cnt,bar = 50)