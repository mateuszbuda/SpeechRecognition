function [ preemph ] = preemp( input, p )
%PREEMP Pre-emphasis filter.
%     Args:
%         input: array of speech frames [N x M] where N is the number of frames and
%                M the samples per frame
%         p: preemhasis factor (defaults to the value specified in the exercise)
% 
%     Output:
%         array of pre-emphasised speech samples

if nargin > 2
    error('preemp:TooManyInputs', ...
        strcat('Args:\n', ...
        'input: array of speech frames [N x M] where N is the number of frames and M the samples per frame\n', ...
        'p: preemhasis factor (defaults to the value specified in the exercise)\n'));
end

switch nargin
    case 1
        p = 0.97;
end


% This should work but it doesn't. Why?
% preemph = filter([1, -p], 1, input);

preemph(:, 1) = input(:, 1);

for i = size(input, 2):-1:2
    
    preemph(:, i) = input(:, i) - (p .* input(:, i - 1));
    
end
