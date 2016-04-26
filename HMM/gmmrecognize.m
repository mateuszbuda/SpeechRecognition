function [ error ] = gmmrecognize( models, utterances, hmm )
%GMMRECOGNIZE Computes log likelihood of each model for each utterance and
%matches best model. Outputs recognition error rate.

if nargin < 3
    hmm = 0;
end

N = size(utterances, 2);
M = size(models, 2);

logliks = zeros(N, M);

for n = 1:N
    
    mfcc = utterances{n}.mfcc;
    
    for m = 1:M
        
        model = models{m};
        
        if hmm == 0
            means = model.gmm.means;
            covars = model.gmm.covars;
            weights = model.gmm.weights;
        else
            means = model.hmm.means;
            covars = model.hmm.covars;
            weights = ones(size(model.gmm.weights)) / length(model.gmm.weights);
        end
        
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
