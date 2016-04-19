clc
clear

models = load('lab2_models.mat');
models = models.models;

example{1} = load('lab2_example.mat');

mfcc = example{1}.mfcc;

figure
imagesc(flip(mfcc'))

hmm_obsloglik = logmvndd(mfcc, models{1}.hmm.means, models{1}.hmm.covars);

figure
imagesc(flip(hmm_obsloglik'))

