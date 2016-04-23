function [ viterbi_loglik, viterbi_path ] = viterbi( log_emlik, log_startprob, log_transmat )
%VIRTEBI Viterbi path.
%     Args:
%         log_emlik: NxM array of emission log likelihoods, N frames, M states
%         log_startprob: log probability to start in state i
%         log_transmat: log transition probability from state i to j
% 
%     Output:
%         viterbi_loglik: log likelihood of the best path
%         viterbi_path: best path

N = size(log_emlik, 1);
M = size(log_emlik, 2);

logliks = zeros(N, M) - Inf;
states_trace = zeros(N, M);

logliks(1, :) = log_startprob + log_emlik(1, :);

for n = 2:N
    
    for j = 1:M
        
        [v, i] = max(logliks(n - 1, :) + log_transmat(:, j)');
        
        logliks(n, j) = v + log_emlik(n, j);
        states_trace(n, j) = i;
    
    end
    
end

viterbi_loglik = max(logliks(N, :));

viterbi_path = zeros(1, N);
[~, viterbi_path(N)] = max(states_trace(N, :));

for i = (N - 1):-1:1
    
    viterbi_path(i) = states_trace(i + 1, viterbi_path(i + 1));
    
end

