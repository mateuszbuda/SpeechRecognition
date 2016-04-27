function [ loglik ] = gmmloglik( log_emlik, weights )
%GMMLOGLIK Log Likelihood for a GMM model based on Multivariate Normal
%Distribution.
%     Args:
%         log_emlik: array like, shape (N, K).
%             contains the log likelihoods for each of N observations and
%             each of K distributions
%         weights:   weight vector for the K components in the mixture
% 
%     Output:
%         gmmloglik: scalar, log likelihood of data given the GMM model.

loglik = sum(log(exp(log_emlik) * weights'));

