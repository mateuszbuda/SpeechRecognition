
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



    https://www.pdc.kth.se/resources/software/login-1/macintosh

    cd workdir; \
    tar cpf - dataset | ssh buda@tegner.pdc.kth.se "tar xpf - -C /afs/pdc.kth.se/home/b/buda"
    tar cpf - testset | ssh buda@tegner.pdc.kth.se "tar xpf - -C /cfs/klemming/nobackup/b/buda"


    THEANO_FLAGS='device=gpu0' python $PDNNDIR/cmds/run_DNN.py \
    --train-data "/cfs/klemming/nobackup/b/buda/dataset/train_tr_FBANK.pfile,partition=600m,stream=false,random=true" \
    --valid-data "/cfs/klemming/nobackup/b/buda/dataset/train_va_FBANK.pfile,partition=600m,stream=false,random=true" \
    --nnet-spec "40:1024:1024:1024:1024:64" \
    --activation rectifier \
    --wdir ./nnet1 \
    --param-output-file nnet1.mdl \
    --cfg-output-file nnet1.cfg \
    |& tee -a nnet1.log

    THEANO_FLAGS='device=gpu1' python $PDNNDIR/cmds/run_DNN.py \
    --train-data "/cfs/klemming/nobackup/b/buda/dataset/train_tr_FBANK.pfile,context=5,partition=600m,stream=false,random=true" \
    --valid-data "/cfs/klemming/nobackup/b/buda/dataset/train_va_FBANK.pfile,context=5,partition=600m,stream=false,random=true" \
    --nnet-spec "440:1024:1024:1024:1024:64" \
    --activation rectifier \
    --wdir ./nnet2 \
    --param-output-file nnet2.mdl \
    --cfg-output-file nnet2.cfg \
    |& tee -a nnet2.log

    THEANO_FLAGS='device=gpu0' python $PDNNDIR/cmds/run_DNN.py \
    --train-data "/cfs/klemming/nobackup/b/buda/dataset/train_tr_FBANK.pfile,context=5,partition=600m,stream=false,random=true" \
    --valid-data "/cfs/klemming/nobackup/b/buda/dataset/train_va_FBANK.pfile,context=5,partition=600m,stream=false,random=true" \
    --nnet-spec "440:256:256:256:256:64" \
    --activation rectifier \
    --wdir ./nnet3 \
    --param-output-file nnet3.mdl \
    --cfg-output-file nnet3.cfg \
    |& tee -a nnet3.log

    THEANO_FLAGS='device=gpu1' python $PDNNDIR/cmds/run_DNN.py \
    --train-data "/cfs/klemming/nobackup/b/buda/dataset/train_tr_MFCC_0_D_A.pfile,partition=600m,stream=false,random=true" \
    --valid-data "/cfs/klemming/nobackup/b/buda/dataset/train_va_MFCC_0_D_A.pfile,partition=600m,stream=false,random=true" \
    --nnet-spec "39:1024:1024:1024:1024:64" \
    --activation rectifier \
    --wdir ./nnet4 \
    --param-output-file nnet4.mdl \
    --cfg-output-file nnet4.cfg \
    |& tee -a nnet4.log



    THEANO_FLAGS='device=gpu0' python $PDNNDIR/cmds/run_Extract_Feats.py \
    --data "/cfs/klemming/nobackup/b/buda/testset/test_FBANK.pfile,partition=600m,stream=false,random=true" \
    --nnet-param nnet1.mdl \
    --nnet-cfg nnet1.cfg \
    --output-file "nnet1.classify.pickle.gz" \
    --layer-index -1 \
    --batch-size 128

    THEANO_FLAGS='device=gpu1' python $PDNNDIR/cmds/run_Extract_Feats.py \
    --data "/cfs/klemming/nobackup/b/buda/testset/test_FBANK.pfile,context=5,partition=600m,stream=false,random=false" \
    --nnet-param nnet2.mdl \
    --nnet-cfg nnet2.cfg \
    --output-file "nnet2.classify.pickle.gz" \
    --layer-index -1 \
    --batch-size 128

    THEANO_FLAGS='device=gpu0' python $PDNNDIR/cmds/run_Extract_Feats.py \
    --data "/cfs/klemming/nobackup/b/buda/testset/test_FBANK.pfile,context=5,partition=600m,stream=false,random=true" \
    --nnet-param nnet3.mdl \
    --nnet-cfg nnet3.cfg \
    --output-file "nnet3.classify.pickle.gz" \
    --layer-index -1 \
    --batch-size 128

    THEANO_FLAGS='device=gpu1' python $PDNNDIR/cmds/run_Extract_Feats.py \
    --data "/cfs/klemming/nobackup/b/buda/testset/test_MFCC_0_D_A.pfile,partition=600m,stream=false,random=true" \
    --nnet-param nnet4.mdl \
    --nnet-cfg nnet4.cfg \
    --output-file "nnet4.classify.pickle.gz" \
    --layer-index -1 \
    --batch-size 128


    ssh buda@tegner.pdc.kth.se "tar czpf - /afs/pdc.kth.se/home/b/buda/nnet2" | tar xzpf - -C .


    python tools/evaluate.py


    frame err = 0.144886032479
    phone err = 0.0896678092015


