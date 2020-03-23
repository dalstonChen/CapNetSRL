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

def Conll_print(annotated_sentence):
    for line in annotated_sentence:
        lll = []
        for key, value in line.items():
            lll.append(value)
        print("\t".join(lll))
gold_files = []
baseline_files = []
for i in range(70):
    gold_files.append(open(gold_path+'_sentencelen_'+str(i), 'w'))
    baseline_files.append(open(baseline_path+'_sentencelen_'+str(i), 'w'))

with open(gold_path, encoding="utf8", errors='ignore') as gold_file, \
    open(baseline_path, encoding="utf8", errors='ignore') as baseline_file:
    #annotated_sentence, directed_arc_indices, arc_tags , predicates_indexes, sentence_blob
    sentence_len = [0] * 70
    for gold, baseline in zip(lazy_parse(gold_file.read()), lazy_parse(baseline_file.read())):
        file_id = int(len(gold[4].split('\n')) / 10)
        gold_files[file_id].write(gold[4]+'\n\n')
        baseline_files[file_id].write(baseline[4]+'\n\n')
        sentence_len[file_id] += len(gold[3])

for i in range(70):
    gold_files[i].close()
    baseline_files[i].close()

import os
myCmd = []
for i in range(7):
    myCmd.append(os.popen('perl /afs/inf.ed.ac.uk/group/project/xchen13/scorer/conll2009/eval09.pl -g %s_sentencelen_%d \
-s %s_sentencelen_%d'%(gold_path, i, baseline_path, i)).read())
    print('len = %d'%i)
    print('Num of propostions = %d' % sentence_len[i])
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
for i in range(7):
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
        print(str(i*10)+"-"+str(i*10+9) + '\t' + f)
print([[idx, value] for idx, value in enumerate(sentence_len)])