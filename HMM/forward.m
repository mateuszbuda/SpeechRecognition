function [ forward_prob ] = forward( log_emlik, log_startprob, log_transmat )
%FORWARD Forward probabilities in log domain.
%     Args:
%         log_emlik: NxM array of emission log likelihoods, N frames, M states
%         log_startprob: log probability to start in state i
%         log_transmat: log transition probability from state i to j
% 
%     Output:
%         forward_prob: NxM array of forward log probabilities for each of the M states in the model

N = size(log_emlik, 1);
M = size(log_emlik, 2);

forward_prob = zeros(N, M) - Inf;

forward_prob(1, :) = log_startprob + log_emlik(1, :);

for n = 2:N
    
    for j = 1:M
        
        forward_prob(n, j) = logsumexp(forward_prob(n - 1, :) + log_transmat(:, j)') + log_emlik(n, j);
    
    end
    
end

