function [ lpr ] = logmvndd( X, means, covars )
%LOGMVNDD Compute Gaussian log-density at X for a diagonal model.
%MatLab equivalent to log_multivariate_normal_density_diag in scikit-learn
%   Args:
%       X: array like, shape (n_observations, n_features)
%       means: array like, shape (n_components, n_features)
%       covars: array like, shape (n_components, n_features)
% 
%	Output:
%       lpr: array like, shape (n_observations, n_components)

[n_samples, n_dim] = size(X);

lpr = -0.5 * (repmat(n_dim * log(2 * pi) + sum(log(covars), 2)', n_samples, 1) ...
    + repmat(sum((means .^ 2) ./ covars, 2)', n_samples, 1) ...
    - 2 * (X * (means ./ covars)') ...
    + (X .^ 2) * (1 ./ covars)');

