function [ error ] = hmmrecognize( models, utterances )
%HMMRECOGNIZE Computes log likelihood of each model for each utterance and
%matches best model. Outputs recognition error rate.

N = size(utterances, 2);
M = size(models, 2);

logliks = zeros(N, M);

for n = 1:N
    
    mfcc = utterances{n}.mfcc;
    
    for m = 1:M
        
        model = models{m};
        means = model.hmm.means;
        covars = model.hmm.covars;
        log_startprob = log(model.hmm.startprob);
        log_transmat = log(model.hmm.transmat);
        
        hmm_obsloglik = logmvndd(mfcc, means, covars);
        hmm_logalpha = forward(hmm_obsloglik, log_startprob, log_transmat);
        logliks(n, m) = hmmloglik(hmm_logalpha);
        
    end
    
end

[~, I] = max(logliks, [], 2);

error = 0;

for n = 1:N
    
    if utterances{n}.digit ~= models{I(n)}.digit
        
        error = error + 1;
        
    end
    
end

error = error / N;

