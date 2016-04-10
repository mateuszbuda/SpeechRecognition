function [ mspec ] = logMelSpectrum( input, samplingrate )
%LOGMELSPECTRUM Calculates the log output of a Mel filterbank when the
%input is the power spectrum
%     Args:
%         input: array of power spectrum coefficients [N x nfft] where N is the number of frames and
%                nfft the length of each spectrum
%         samplingrate: sampling rate of the original signal (used to calculate the filterbanks)
%     Output:
%         array of Mel filterbank log outputs [N x nmelfilters] where nmelfilters is the number
%         of filters in the filterbank

nfft = 512;
nfilt = 40;

pytrifbank = py.tools.trfbank(samplingrate, nfft);
trifbank = reshape(double(pytrifbank), nfft, nfilt);

input = input(:, 1:nfft);

mspec = log(input * trifbank);
