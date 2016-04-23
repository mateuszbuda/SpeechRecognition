function [ loglik ] = hmmloglik( hmm_logalpha )
%HMMLOGLIK Log Likelihood for a HMM model
%     Args:
%         hmm_logalpha: NxM array of forward log probabilities for each of the M states in the model
% 
%     Output:
%         hmmloglik: scalar, log likelihood of data given the log alpha.

loglik = logsumexp(hmm_logalpha(end, :));

