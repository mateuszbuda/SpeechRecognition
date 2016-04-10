function [ lmfcc, mfcc, mspec, spec, windowed, preemph, frames ] = mfcc( samples, winlen, winshift, nfft, nceps, samplingrate, liftercoeff )
%MFCC Computes Mel Frequency Cepstrum Coefficients.
%   Args:
%         samples: array of speech samples with shape (N,)
%         winlen: lenght of the analysis window
%         winshift: number of samples to shift the analysis window at every time step
%         nfft: length of the Fast Fourier Transform (power of 2, >= winlen)
%         nceps: number of cepstrum coefficients to compute
%         samplingrate: sampling rate of the original signal
%         liftercoeff: liftering coefficient used to equalise scale of MFCCs
%   Returns:
%         N x nceps array with lifetered MFCC coefficients

if nargin > 7
    error('mfcc:TooManyInputs', ...
        strcat('Args:\n', ...
        'samples: array of speech samples with shape (N,)\n', ...
        'winlen: lenght of the analysis window\n', ...
        'winshift: number of samples to shift the analysis window at every time step\n', ...
        'nfft: length of the Fast Fourier Transform (power of 2, >= winlen)\n', ...
        'nceps: number of cepstrum coefficients to compute\n', ...
        'samplingrate: sampling rate of the original signal\n', ...
        'liftercoeff: liftering coefficient used to equalise scale of MFCCs\n'));
end

switch nargin
    case 1
        winlen = 400;
        winshift = 200;
        nfft = 512;
        nceps = 13;
        samplingrate = 20000;
        liftercoeff = 22;
        
    case 2
        winshift = 200;
        nfft = 512;
        nceps = 13;
        samplingrate = 20000;
        liftercoeff = 22;
        
    case 3
        nfft = 512;
        nceps = 13;
        samplingrate = 20000;
        liftercoeff = 22;
        
    case 4
        nceps = 13;
        samplingrate = 20000;
        liftercoeff = 22;
        
    case 5
        samplingrate = 20000;
        liftercoeff = 22;
        
    case 6
        liftercoeff = 22;
end


frames = enframe(samples, winlen, winshift);

preemph = preemp(frames, 0.97);

windowed = windowing(preemph);

spec = powerSpectrum(windowed, nfft);

mspec = logMelSpectrum(spec, samplingrate);

mfcc = cepstrum(mspec, nceps);

lmfcc = lifter(mfcc, liftercoeff)';
