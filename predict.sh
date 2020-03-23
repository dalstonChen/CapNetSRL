allennlp predict  /disk/scratch1/xchen13/conll09_prop/model.tar.gz \
/afs/inf.ed.ac.uk/group/project/xchen13/datasets/american_news/extract_copy/src_conll09_format.instances_only \
--batch-size 128 --cuda-device 0 --use-dataset-reader --include-package myallennlp --predictor dependency_srl