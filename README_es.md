allennlp predict  /disk/scratch1/xchen13/es_advanced_iter2_argcaplinear_global_conll09_100ep/model.tar.gz \
/afs/inf.ed.ac.uk/group/project/xchen13/datasets/2009_conll_p1/data/CoNLL2009-ST-Spanish/CoNLL2009-ST-Spanish-development.txt \
--batch-size 32 --cuda-device 1  --use-dataset-reader --include-package myallennlp --predictor dependency_srl 

allennlp predict  /disk/scratch1/xchen13/es_advanced_iter2_argcaplinear_global_conll09_100ep/model.tar.gz \
/afs/inf.ed.ac.uk/group/project/xchen13/datasets/2009_conll_p1/data/CoNLL2009-ST-Spanish/CoNLL2009-ST-evaluation-Spanish.txt \
--batch-size 32 --cuda-device 2  --use-dataset-reader --include-package myallennlp --predictor dependency_srl 



 perl /afs/inf.ed.ac.uk/group/project/xchen13/scorer/conll2009/eval09.pl -g /afs/inf.ed.ac.uk/group/project/xchen13/datasets/2009_conll_p1/data/CoNLL2009-ST-Spanish/CoNLL2009-ST-Spanish-development.txt \
-s /disk/scratch1/xchen13/es_advanced_iter2_argcaplinear_global_conll09_100ep/conll09_dev.predict


 perl /afs/inf.ed.ac.uk/group/project/xchen13/scorer/conll2009/eval09.pl -g /afs/inf.ed.ac.uk/group/project/xchen13/datasets/2009_conll_p1/data/CoNLL2009-ST-Spanish/CoNLL2009-ST-evaluation-Spanish.txt \
-s /disk/scratch1/xchen13/es_advanced_iter2_argcaplinear_global_conll09_100ep/conll09_test.predict