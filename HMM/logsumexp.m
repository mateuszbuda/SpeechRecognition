function s = logsumexp(X, dim)
%LOGSUMEXP Compute log(sum(exp(X),dim)) while avoiding numerical underflow.
%   By default dim = 1 (columns).
% Written by Mo Chen (sth4nth@gmail.com).

if nargin == 1, 
    % Determine which dimension sum will use
    dim = find(size(X) ~=1 , 1);
    if isempty(dim), dim = 1; end
end

% subtract the largest in each dim
y = max(X, [], dim);
s = y + log(sum(exp(bsxfun(@minus, X, y)), dim));
i = isinf(y);
if any(i(:))
    s(i) = y(i);
end

