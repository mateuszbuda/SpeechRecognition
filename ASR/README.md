
awk '/\/man/' train.lst > train_man.lst
awk '/\/woman/' train.lst > train_woman.lst

awk '/\/a/' train_man.lst > train_va.lst
awk '/\/a/' train_woman.lst >> train_va.lst

awk '!/\/a/' train_man.lst > train_tr.lst
awk '!/\/a/' train_woman.lst >> train_tr.lst

wc -l train.lst
wc -l train_man.lst
wc -l train_woman.lst
wc -l train_tr.lst
wc -l train_va.lst



tools/htk2pfile.py workdir/train_tr_align.mlf workdir/state2id.lst FBANK workdir/train_tr_FBANK.pfile
tools/htk2pfile.py workdir/train_va_align.mlf workdir/state2id.lst FBANK workdir/train_va_FBANK.pfile
tools/htk2pfile.py workdir/test_align.mlf workdir/state2id.lst FBANK workdir/test_FBANK.pfile

tools/htk2pfile.py workdir/train_tr_align.mlf workdir/state2id.lst MFCC_0_D_A workdir/train_tr_MFCC_0_D_A.pfile
tools/htk2pfile.py workdir/train_va_align.mlf workdir/state2id.lst MFCC_0_D_A workdir/train_va_MFCC_0_D_A.pfile
tools/htk2pfile.py workdir/test_align.mlf workdir/state2id.lst MFCC_0_D_A workdir/test_MFCC_0_D_A.pfile


pfile_norm -mean 0 -std 1 -i workdir/train_tr_FBANK.pfile -o workdir/train_tr_FBANK_n.pfile
pfile_norm -mean 0 -std 1 -i workdir/train_va_FBANK.pfile -o workdir/train_va_FBANK_n.pfile
pfile_norm -mean 0 -std 1 -i workdir/test_FBANK.pfile -o workdir/test_FBANK_n.pfile

pfile_norm -mean 0 -std 1 -i workdir/train_tr_MFCC_0_D_A.pfile -o workdir/train_tr_MFCC_0_D_A_n.pfile
pfile_norm -mean 0 -std 1 -i workdir/train_va_MFCC_0_D_A.pfile -o workdir/train_va_MFCC_0_D_A_n.pfile
pfile_norm -mean 0 -std 1 -i workdir/test_MFCC_0_D_A.pfile -o workdir/test_MFCC_0_D_A_n.pfile


rm test_FBANK.pfile test_MFCC_0_D_A.pfile train_tr_FBANK.pfile train_tr_MFCC_0_D_A.pfile train_va_FBANK.pfile train_va_MFCC_0_D_A.pfile

mv test_FBANK_n.pfile test_FBANK.pfile 
mv test_MFCC_0_D_A_n.pfile test_MFCC_0_D_A.pfile 
mv train_tr_FBANK_n.pfile train_tr_FBANK.pfile 
mv train_tr_MFCC_0_D_A_n.pfile train_tr_MFCC_0_D_A.pfile 
mv train_va_FBANK_n.pfile train_va_FBANK.pfile 
mv train_va_MFCC_0_D_A_n.pfile train_va_MFCC_0_D_A.pfile



python $PDNNDIR/cmds/run_DNN.py \
--train-data "workdir/train_tr_FBANK.pfile,partition=600m,stream=false,random=true" \
--valid-data "workdir/train_va_FBANK.pfile,partition=600m,stream=false,random=true" \
--nnet-spec "40:1024:1024:1024:1024:64" \
--activation rectifier \
--wdir ./ \
--param-output-file nnet1.mdl \
--cfg-output-file nnet1.cfg \
|& tee -a nnet1.log

python $PDNNDIR/cmds/run_DNN.py \
--train-data "workdir/train_tr_FBANK.pfile,context=5,partition=600m,stream=false,random=true" \
--valid-data "workdir/train_va_FBANK.pfile,context=5,partition=600m,stream=false,random=true" \
--nnet-spec "440:1024:1024:1024:1024:64" \
--activation rectifier \
--wdir ./ \
--param-output-file nnet2.mdl \
--cfg-output-file nnet2.cfg \
|& tee -a nnet2.log

python $PDNNDIR/cmds/run_DNN.py \
--train-data "workdir/train_tr_FBANK.pfile,context=5,partition=600m,stream=false,random=true" \
--valid-data "workdir/train_va_FBANK.pfile,context=5,partition=600m,stream=false,random=true" \
--nnet-spec "440:256:256:256:256:64" \
--activation rectifier \
--wdir ./ \
--param-output-file nnet3.mdl \
--cfg-output-file nnet3.cfg \
|& tee -a nnet3.log

python $PDNNDIR/cmds/run_DNN.py \
--train-data "workdir/train_tr_MFCC_0_D_A.pfile,partition=600m,stream=false,random=true" \
--valid-data "workdir/train_va_MFCC_0_D_A.pfile,partition=600m,stream=false,random=true" \
--nnet-spec "40:1024:1024:1024:1024:64" \
--activation rectifier \
--wdir ./ \
--param-output-file nnet4.mdl \
--cfg-output-file nnet4.cfg \
|& tee -a nnet4.log
