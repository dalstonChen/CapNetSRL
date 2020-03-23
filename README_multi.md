


Dev set:

allennlp predict  /disk/scratch1/xchen13/single_iter2_weight_conll09_multi_both_600ep/model.tar.gz \
/afs/inf.ed.ac.uk/group/project/xchen13/datasets/2009_conll_p2/data/CoNLL2009-ST-English/CoNLL2009-ST-English-development.txt \
--batch-size 32 --cuda-device 3  --use-dataset-reader --include-package myallennlp --predictor dependency_srl 

Test set:
 
allennlp predict  /disk/scratch1/xchen13/single_iter2_weight_conll09_global_600ep/model.tar.gz \
/afs/inf.ed.ac.uk/group/project/xchen13/datasets/2009_conll_p2/data/CoNLL2009-ST-English/CoNLL2009-ST-evaluation-English.txt \
--batch-size 32 --cuda-device 0  --use-dataset-reader --include-package myallennlp --predictor dependency_srl 

OOD set:

allennlp predict  /disk/scratch1/xchen13/conll09_all_hinge_ce/model.tar.gz \
/afs/inf.ed.ac.uk/group/project/xchen13/datasets/2009_conll_p2/data/CoNLL2009-ST-English/CoNLL2009-ST-evaluation-English-ood.txt \
--batch-size 32 --cuda-device 1  --use-dataset-reader --include-package myallennlp --predictor dependency_srl 


Evaluate



 perl /afs/inf.ed.ac.uk/group/project/xchen13/scorer/conll2009/eval09.pl -g /afs/inf.ed.ac.uk/group/project/xchen13/datasets/2009_conll_p2/data/CoNLL2009-ST-English/CoNLL2009-ST-English-development.txt \
-s /disk/scratch1/xchen13/single_iter2_weight_conll09_multi_both_600ep/conll09_dev.predict


 perl /afs/inf.ed.ac.uk/group/project/xchen13/scorer/conll2009/eval09.pl -g /afs/inf.ed.ac.uk/group/project/xchen13/datasets/2009_conll_p2/data/CoNLL2009-ST-English/CoNLL2009-ST-evaluation-English.txt \
-s /disk/scratch1/xchen13/single_iter2_weight_conll09_global_600ep/conll09_test.predict

 perl /afs/inf.ed.ac.uk/group/project/xchen13/scorer/conll2009/eval09.pl -g /afs/inf.ed.ac.uk/group/project/xchen13/datasets/2009_conll_p2/data/CoNLL2009-ST-English/CoNLL2009-ST-evaluation-English-ood.txt \
-s /disk/scratch1/xchen13/conll09_all_hinge_ce/conll09_ood.predict