function [ error ] = gmmrecognize( models, utterances )
%GMMRECOGNIZE Computes log likelihood of each model for each utterance and
%matches best model. Outputs recognition error rate.

N = size(utterances, 2);
M = size(models, 2);

logliks = zeros(N, M);

for n = 1:N
    
    mfcc = utterances{n}.mfcc;
    
    for m = 1:M
        
        model = models{m};
        means = model.gmm.means;
        covars = model.gmm.covars;
        weights = model.gmm.weights;
        
        gmm_obsloglik = logmvndd(mfcc, means, covars);
        logliks(n, m) = gmmloglik(gmm_obsloglik, weights);
        
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
