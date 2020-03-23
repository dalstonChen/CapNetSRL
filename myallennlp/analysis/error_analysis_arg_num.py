import sys
sys.path.append("/afs/inf.ed.ac.uk/user/x/xchen13/project/SRL/")

from myallennlp.dataset_readers.conll2009 import lazy_parse
from allennlp.common.file_utils import cached_path

gold_path = '/afs/inf.ed.ac.uk/group/project/xchen13/datasets/2009_conll_p2/data/CoNLL2009-ST-English/CoNLL2009-ST-evaluation-English.txt'
#capsule_path = '/disk/scratch1/xchen13/single_iter2_weight_conll09_600ep/conll09_test.predict'
#baseline_path = '/disk/scratch1/xchen13/single_ave_conll09_600ep/conll09_test.predict'
baseline_path = '/disk/scratch1/xchen13/single_iter2_weight_conll09_global_600ep/conll09_test.predict'
#capsule_path = '/disk/scratch1/xchen13/conll09_all_hinge_ce/conll09_test.predict'
gold_path = cached_path(gold_path)
#capsule_path = cached_path(capsule_path)
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
            cnt += 1
            print(predicates[arc_indice[1]],arc_indice[1])
            multi_ans.append((arc_indice[0], arc_tag))
    return cnt, multi_ans

def Conll_print(f, sentence_blob, pred_idx, pred_position):
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
    f.write("\n".join(res) + '\n\n')


gold_files = []
baseline_files = []
for i in range(8):
    gold_files.append(open(gold_path+'_argnum_'+str(i), 'w'))
    baseline_files.append(open(baseline_path+'_argnum_'+str(i), 'w'))

with open(gold_path, encoding="utf8", errors='ignore') as gold_file, \
    open(baseline_path, encoding="utf8", errors='ignore') as baseline_file:
    #annotated_sentence, directed_arc_indices, arc_tags , predicates_indexes, sentence_blob
    arg_nums = [0] * 8
    for gold, baseline in zip(lazy_parse(gold_file.read()), lazy_parse(baseline_file.read())):
        #file_id = int(len(gold[0].split('\n')) / 10)
        #gold_files[file_id].write(gold[0]+'\n\n')
        #baseline_files[file_id].write(baseline[0]+'\n\n')
        pred2args = {}
        for (idx, pred_idx), tag in zip(gold[1],gold[2]):
            if pred2args.get(pred_idx) == None: pred2args[pred_idx] = []
            pred2args[pred_idx].append((idx, tag))
        #print(gold[0])
        for k,v in pred2args.items():
            arg_nums[len(v)] += 1
            #print(k,v)
            Conll_print(gold_files[len(v)], gold[4], k, gold[3][k])
            Conll_print(baseline_files[len(v)], baseline[4], k, baseline[3][k])
    print([[idx, value] for idx, value in enumerate(arg_nums)])

for i in range(8):
    gold_files[i].close()
    baseline_files[i].close()

import os
myCmd = []
for i in range(8):
    myCmd.append(os.popen('perl /afs/inf.ed.ac.uk/group/project/xchen13/scorer/conll2009/eval09.pl -g %s_argnum_%d \
-s %s_argnum_%d'%(gold_path, i, baseline_path, i)).read())
    print('arg num = %d'%i)
    print('Num of propostions = %d' % arg_nums[i])
    if (len(myCmd[-1].split('\n'))<8): print('None')
    else:
        print(myCmd[-1].split('\n')[7])
        print(myCmd[-1].split('\n')[8])
        print(myCmd[-1].split('\n')[9])
        print(myCmd[-1].split('\n')[15])
P = []
R = []
F = []
EM = []
for i in range(8):
    if (len(myCmd[i].split('\n'))<8): continue
    else:
        p = myCmd[i].split('\n')[7][-7:-2]
        r = myCmd[i].split('\n')[8][-7:-2]
        f = myCmd[i].split('\n')[9][-6:]
        em = myCmd[i].split('\n')[15][-6:]
        #print (i, p,r,f,em)
        P.append((i, p))
        R.append((i, r))
        F.append((i, f))
        EM.append((i, em))
        print(str(i) + '\t' + f)
print([[idx, value] for idx, value in enumerate(arg_nums)])
