allennlp predict  /disk/scratch1/xchen13/de_single_iter2_weight_conll09_global_200ep/model.tar.gz \
/afs/inf.ed.ac.uk/group/project/xchen13/datasets/2009_conll_p1/data/CoNLL2009-ST-German/CoNLL2009-ST-German-development.txt \
--batch-size 32 --cuda-device 3  --use-dataset-reader --include-package myallennlp --predictor dependency_srl 


allennlp predict  /disk/scratch1/xchen13/de_single_iter2_weight_conll09_global_100ep/model.tar.gz \
/afs/inf.ed.ac.uk/group/project/xchen13/datasets/2009_conll_p1/data/CoNLL2009-ST-German/CoNLL2009-ST-evaluation-German.txt \
--batch-size 16 --cuda-device 0  --use-dataset-reader --include-package myallennlp --predictor dependency_srl 



 perl /afs/inf.ed.ac.uk/group/project/xchen13/scorer/conll2009/eval09.pl -g /afs/inf.ed.ac.uk/group/project/xchen13/datasets/2009_conll_p1/data/CoNLL2009-ST-German/CoNLL2009-ST-German-development.txt \
-s /disk/scratch1/xchen13/de_single_iter2_weight_conll09_global_200ep/conll09_dev.predict


 perl /afs/inf.ed.ac.uk/group/project/xchen13/scorer/conll2009/eval09.pl -g /afs/inf.ed.ac.uk/group/project/xchen13/datasets/2009_conll_p1/data/CoNLL2009-ST-German/CoNLL2009-ST-evaluation-German.txt \
-s /disk/scratch1/xchen13/de_single_iter2_weight_conll09_global_100ep/conll09_test.predict