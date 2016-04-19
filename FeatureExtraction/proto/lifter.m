function [ lmfcc ] = lifter( mfcc, lifternum )
%LIFTER Applies liftering to improve the relative range of MFCC coefficients.
%   Args:
%        mfcc: NxM matrix where N is the number of frames and M the number of MFCC coefficients
%        lifternum: lifering coefficient
% 
%     Returns:
%        NxM array with lifeterd coefficients

if nargin > 2
    error('lifter:TooManyInputs', ...
        strcat('Args:\n', ...
        'mfcc_out: NxM matrix where N is the number of frames and M the number of MFCC coefficients\n', ...
        'lifternum: lifering coefficient\n'));
end

switch nargin
    case 1
        lifternum = 22;
end

[nceps, nframes] = size(mfcc');

cepwin = 1.0 + lifternum / 2.0 * sin(pi * (0:nceps-1)' / lifternum);
lmfcc = (mfcc' .* (repmat(cepwin, 1, nframes)));
