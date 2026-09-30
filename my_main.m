
addpath(genpath('.\UDCMVFC-main\'));
load('MSRC.mat','X','Y'); y = Y;   beta=0.9;tau=0.4;a=0.004;r=1.1;l=1.2;

%load('mul_ORL.mat','data','labels'); X = data; y = labels; beta=0.9;tau=0.2;a=0.005;r=1.5;l=1.7; 

% load('RGB-D.mat','X','y');   beta=0.9;tau=0.2;a=0.005;r=1.3;l=1.0;   

c = max(y);
nv = length(X); 
for ni = 1:nv
    X{ni} = mapminmax(X{ni}', 0, 1);  
end 
X = cellfun(@(x) x', X, 'UniformOutput', false);
max_iters = 100; tolerance = 1e-4;

tic
[Y_pred,obj_history] = UDCMVFC(X,c,max_iters,tolerance,r,a,tau,l,beta);
time = toc;
[~, label_out] = max(Y_pred, [], 2); 
result_cluster = ClusteringMeasure(y, label_out); 
acc= result_cluster(1);
purity= result_cluster(2);
nmi = result_cluster(3);
ARI= result_cluster(4);