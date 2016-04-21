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

%% exapmle
mfcc = example{1}.mfcc;

if plot_mfcc
    figure
    imagesc(flip(mfcc'))
    title('mfcc: MFCC')
end

gmm_obsloglik = logmvndd(mfcc, models{1}.gmm.means, models{1}.gmm.covars);

if plot_gmm_obsloglik
    figure
    imagesc(flip(gmm_obsloglik'))
    title('gmm_obsloglik: GMM component/observation log likelihood')
end

hmm_obsloglik = logmvndd(mfcc, models{1}.hmm.means, models{1}.hmm.covars);

if plot_hmm_obsloglik
    figure
    imagesc(flip(hmm_obsloglik'))
    title('hmm_obsloglik: HMM component/observation log likelihood')
end

%% recognition with GMM

error = gmmrecognize(models, tidigits);

%% cleanup

clear plot_mfcc plot_gmm_obsloglik plot_hmm_obsloglik

