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

%% plotting flags

plot_sample = 0;
plot_frames = 0;
plot_preemph = 0;
plot_windowed = 0;
plot_spec = 0;
plot_mspec = 0;
plot_mfccoef = 0;
plot_lmfc = 0;
plot_correlations = 0;
plot_distances = 0;
plot_dendrogram = 0;

%% MFCC for example utterance

samples = example{1}.samples;

if plot_sample
    figure
    plot(samples)
    title('samples: speech samples')
end


[ lmfcc, mfccoef, mspec, spec, windowed, preemph, frames ] = mfcc(samples);


if plot_frames
    figure
    imagesc(flip(frames'))
    title('frames: enframed samples')
end

if plot_preemph
    figure
    imagesc(flip(preemph'))
    title('preemph: preemphasis')
end

if plot_windowed
    figure
    imagesc(flip(windowed'))
    title('windowed: hamming window')
end

if plot_spec
    figure
    imagesc(flip(spec(:, 1:end/2)'))
    title('spec: abs(FFT)^2')
end

if plot_mspec
    figure
    imagesc(flip(mspec'))
    title('mspec: Mel filterbank')
end

if plot_mfccoef
    figure
    imagesc(flip(mfccoef'))
    title('mfcc: MFCCs')
end

if plot_lmfc
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

if plot_correlations
    figure
    imagesc(mfcc_correlation_matrix)
    title('mfcc correlation coefficients matrix')
    figure
    imagesc(mspec_correlation_matrix)
    title('mspec correlation coefficients matrix')
end

clear N tidigits_mfcc tidigits_mspec lmfcc mspec mfcc_correlation_matrix mspec_correlation_matrix

%% Comparing Utterances

N = size(tidigits, 2);

mfccs = cell(1, N);

for i = 1:N
    
    mfccs{i} = mfcc(tidigits{i}.samples)';
    
end

distances = zeros(N);

for i = 1:N
    for j = 1:N
        
        distances(i, j) = dtw(mfccs{i}', mfccs{j}');
        
    end
end

if plot_distances
    figure
    imagesc(distances)
    title('mfcc distances')
end

clear mfccs

%% hierarchical clustering

d = [];

for i = 1:(N - 1)
    
    d = [d distances(i, i + 1:end)];
    
end

Z = linkage(d, 'complete');

labels = cell(1, N);

for i = 1:N
    
    labels{i} = [tidigits{i}.gender '-' tidigits{i}.speaker '-' tidigits{i}.digit '-' tidigits{i}.repetition];
    
end

if plot_dendrogram
    figure
    dendrogram(Z, 0, 'Labels', labels, 'Orientation', 'left', 'ColorThreshold', 'default');
end

clear d labels Z

%% cleanup

clear P N i j samples
clear plot_correlations plot_dendrogram plot_distances plot_frames plot_lmfc ...
    plot_lmfc plot_mfccoef plot_mspec plot_preemph plot_sample plot_spec plot_windowed
