clc;
clear;

addpath(genpath('D:\.\UDCMVFC-main\'));   
load('MSRC.mat','X','Y');
y=Y;


%load('mul_ORL.mat','data','labels');
%X = data;
%y = labels;    %beta=0.6;r=1.4;l=1.8;tau=0.3;a=0.002;

c = max(y);

nv = length(X); 
for ni = 1:nv
    X{ni} = mapminmax(X{ni}', 0, 1);  
end 
X = cellfun(@(x) x', X, 'UniformOutput', false);

max_iters = 100; tolerance = 1e-4;

beta=0.6;
r=1.1;l=1.3;
tau=0.5;a=0.005;

[Y,Yv, obj_history] = UDCMVFC(X, c,max_iters,tolerance,r,a,tau,l,beta);
[~, label_out] = max(Y, [], 2);
result_cluster = ClusteringMeasure(y, label_out); 
acc= result_cluster(1);
purity= result_cluster(2);
nmi = result_cluster(3);
ARI= result_cluster(4);