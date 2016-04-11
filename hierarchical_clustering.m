clc
addpath('E:\Master_Machinelearning\Speech recognition\dt2118_lab1_2016-04-01\voicebox')

load tidigits;

winlen_time = 0.020;% 20 ms
shift_time = 0.010;
winlen = 400;
winshift = 200;
nfft=512;
nceps=13;
samplingrate=20000;
liftercoeff=22;



for j=1:44
samples = tidigits{j}.samples;   
MFCC{j}= Mfcc(samples, winlen, winshift, nfft, nceps, samplingrate, liftercoeff);

end

for k=1:44
    for m=1:44
dtw = dtw_distance(MFCC{k}, MFCC{m});
% set the final value of the matrix to the dtw length
d (k,m)= dtw(end,end);
    end
end

y=[];
for k=1:43
    y=[y d(k,k+1:end)];
      
end
   


 Z = linkage(y,'complete');
%  for i=1: 43
%  H = dendrogram(Z,'Labels',tidigits{i}.gender,tidigits{i}.speaker, tidigits{i}.digit,tidigits{i}.repetition );
%  end
H = dendrogram(Z,0);
 

