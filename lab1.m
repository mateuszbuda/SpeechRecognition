%% environment setup

clc
clear

addpath('./proto')

P = py.sys.path;
if count(P, './proto') == 0
    insert(P, int32(0), './proto');
end

load example
load tidigits

%% MFCC for example utterance

samples = example{1}.samples;

if false
    figure
    plot(samples)
    title('samples: speech samples')
end


[ lmfcc, mfccoef, mspec, spec, windowed, preemph, frames ] = mfcc(samples);


if false
    figure
    imagesc(flip(frames'))
    title('frames: enframed samples')
end

if false
    figure
    imagesc(flip(preemph'))
    title('preemph: preemphasis')
end

if false
    figure
    imagesc(flip(windowed'))
    title('windowed: hamming window')
end

if false
    figure
    imagesc(flip(spec(:, 1:end/2)'))
    title('spec: abs(FFT)^2')
end

if false
    figure
    imagesc(flip(mspec'))
    title('mspec: Mel filterbank')
end

if false
    figure
    imagesc(flip(mfccoef'))
    title('mfcc: MFCCs')
end

if false
    figure
    imagesc(lmfcc')
    title('lmfcc: Liftered MFCCs')
end

clear lmfcc mfccoef mspec spec windowed preemph frames

%% MFCC for tidigits dataset

N = size(tidigits, 2);

tidigits_mfcc = [];
tidigits_mspec = [];

for i = 1:N
    
    [ lmfcc, ~, mspec ] = mfcc(tidigits{i}.samples);
    tidigits_mfcc = [tidigits_mfcc; lmfcc];
    tidigits_mspec = [tidigits_mspec; mspec];
    
end

mfcc_correlation_matrix = corrcoef(tidigits_mfcc);
mspec_correlation_matrix  = corrcoef(tidigits_mspec);

if false
    figure
    imagesc(mfcc_correlation_matrix);
    title('mfcc correlation coefficients matrix')
    figure
    imagesc(mspec_correlation_matrix);
    title('mspec correlation coefficients matrix')
end

clear P
