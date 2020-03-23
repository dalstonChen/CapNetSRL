allennlp predict  /disk/scratch1/xchen13/cs_single_ave_conll09_200ep/model.tar.gz \
/afs/inf.ed.ac.uk/group/project/xchen13/datasets/2009_conll_p1/data/CoNLL2009-ST-Czech/CoNLL2009-ST-Czech-development.txt \
--batch-size 8 --cuda-device 3  --use-dataset-reader --include-package myallennlp --predictor dependency_srl 

allennlp predict  /disk/scratch1/xchen13/cs_single_ave_baseline_conll09_100ep/model.tar.gz \
/afs/inf.ed.ac.uk/group/project/xchen13/datasets/2009_conll_p1/data/CoNLL2009-ST-Czech/CoNLL2009-ST-evaluation-Czech.txt \
--batch-size 32 --cuda-device 1  --use-dataset-reader --include-package myallennlp --predictor dependency_srl 



 perl /afs/inf.ed.ac.uk/group/project/xchen13/scorer/conll2009/eval09.pl -g /afs/inf.ed.ac.uk/group/project/xchen13/datasets/2009_conll_p1/data/CoNLL2009-ST-Czech/CoNLL2009-ST-Czech-development.txt \
-s /disk/scratch1/xchen13/cs_single_ave_conll09_200ep/conll09_dev.predict


 perl /afs/inf.ed.ac.uk/group/project/xchen13/scorer/conll2009/eval09.pl -g /afs/inf.ed.ac.uk/group/project/xchen13/datasets/2009_conll_p1/data/CoNLL2009-ST-Czech/CoNLL2009-ST-evaluation-Czech.txt \
-s /disk/scratch1/xchen13/cs_single_ave_baseline_conll09_100ep/conll09_test.predict