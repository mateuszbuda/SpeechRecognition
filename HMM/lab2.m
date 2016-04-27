%% setup, load data
clc
clear

models = load('lab2_models.mat');
models = models.models;

tidigits = load('lab2_tidigits.mat');
tidigits = tidigits.tidigits;

example{1} = load('lab2_example.mat');

%% plotting flags

plot_mfcc = 0;
plot_gmm_obsloglik = 0;
plot_hmm_obsloglik = 0;
plot_hmm_logalpha = 0;

%% exapmle
mfcc = example{1}.mfcc;

if plot_mfcc
    figure('position', [100 400 1000 200])
    imagesc(flip(mfcc'))
    title('mfcc: MFCC')
end

gmm_obsloglik = logmvndd(mfcc, models{1}.gmm.means, models{1}.gmm.covars);

if plot_gmm_obsloglik
    figure('position', [100 400 1000 200])
    imagesc(flip(gmm_obsloglik'))
    title('gmm obsloglik: GMM component/observation log likelihood')
end

hmm_obsloglik = logmvndd(mfcc, models{1}.hmm.means, models{1}.hmm.covars);

if plot_hmm_obsloglik
    figure('position', [100 400 1000 200])
    imagesc(flip(hmm_obsloglik'))
    title('hmm obsloglik: HMM component/observation log likelihood')
end

hmm_logalpha = forward(hmm_obsloglik, log(models{1}.hmm.startprob), log(models{1}.hmm.transmat));

if plot_hmm_logalpha
    figure('position', [100 400 1000 200])
    imagesc(flip(hmm_logalpha'))
    title('hmm logalpha: log alpha')
end

hmm_loglik = hmmloglik(hmm_logalpha);

[hmm_vloglik, hmm_vpath] = viterbi(hmm_obsloglik, log(models{1}.hmm.startprob), log(models{1}.hmm.transmat));

if plot_hmm_logalpha
    hold on
    plot(1:size(hmm_logalpha', 2), hmm_vpath)
    hold off
end

%% recognition

err_gmm = round(gmmrecognize(models, tidigits) * length(tidigits));

err_hmm = round(hmmrecognize(models, tidigits) * length(tidigits));

err_hmm_gmm = round(gmmrecognize(models, tidigits, 1) * length(tidigits));

err_hmm_viterbi = round(hmmrecognize(models, tidigits, 1) * length(tidigits));

%% cleanup

clear plot_mfcc plot_gmm_obsloglik plot_hmm_obsloglik plot_hmm_logalpha

