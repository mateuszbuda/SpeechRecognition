function [ ceps ] = cepstrum( input, nceps )
%CEPSTRUM Calulates Cepstral coefficients from mel spectrum applying
%Discrete Cosine Transform
%   Args:
%         input: array of log outputs of Mel scale filterbank [N x nmelfilters] where N is the
%                number of frames and nmelfilters the length of the filterbank
%         nceps: number of output cepstral coefficients
%   Output:
%         array of Cepstral coefficients [N x nceps]

ceps = dct(input')';
ceps = ceps(:, 1:nceps);
