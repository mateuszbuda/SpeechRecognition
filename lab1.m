clc
clear

addpath('./proto')

P = py.sys.path;
if count(P, './proto') == 0
    insert(P, int32(0), './proto');
end

load example
samples = example{1}.samples;

if false
    figure
    plot(samples)
    title('samples: speech samples')
end


[ lmfcc, mfcc, mspec, spec, windowed, preemph, frames ] = mfcc(samples);


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
    imagesc(flip(mfcc'))
    title('mfcc: MFCCs')
end

if false
    figure
    imagesc(lmfcc')
    title('lmfcc: Liftered MFCCs')
    axis xy
end

clear P
