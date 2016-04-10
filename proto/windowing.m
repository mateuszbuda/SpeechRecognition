function [ windowed ] = windowing( input )
%WINDOWING Applies hamming window to the input frames.
%     Args:
%         input: array of speech samples [N x M] where N is the number of frames and
%                M the samples per frame
%     Output:
%         array of windowed speech samples [N x M]

hammingWindow = hamming(size(input, 2), 'periodic')';
hammingMatrix = repmat(hammingWindow, size(input, 1), 1);
    
windowed = input .* hammingMatrix;
