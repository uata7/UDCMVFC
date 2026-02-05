function [Y,Yv, obj_history] = UDCMVFC(X, c,max_iters,tolerance,r,a,tau,l,beta)
rng(4, 'twister');  
num_views = length(X); 
n = size(X{1}, 1);
Y = rand(n, c);
Y = Y ./ sum(Y, 2); 
M = cell(num_views, 1);
YTT=(Y.^r)'; 
YTTT=sum(YTT,2);
momen = cell(num_views, 1);
for v = 1:num_views
M{v}= YTT*X{v} ./ YTTT;
momen{v} = zeros(size(M{v}));
end
obj_history = zeros(max_iters, 1);
uall = cell(num_views, 1);
normall = cell(num_views, 1); 
d = zeros(num_views, 1);
 for  v = 1:num_views
     dtt = pdist2(X{v}, M{v},'euclidean');
     uall{v} = dtt.^(2/(1-r));
     normall{v} = sqrt(sum(uall{v}.^2, 2)); 
     d(v)=size(M{v}, 2);
 end
for iter = 1:max_iters 
       for v0 = 1:num_views
       dlcc_dm = UDCMVFC_grad(X, M ,uall, normall,v0,r,tau,l);
       momen{v0} = beta * momen{v0} + dlcc_dm;
       M{v0} = M{v0} - a * momen{v0};
       uall{v0}=pdist2(X{v0}, M{v0},'euclidean').^(2/(1-r));
       normall{v0} = sqrt(sum(uall{v0}.^2, 2)); 
       end
UUUalls = vertcat(uall{:});
NNNorms = vertcat(normall{:});
SS = UUUalls * UUUalls'; 
norm_p = NNNorms * NNNorms';
SS = SS ./ norm_p;
SS(1:size(SS,1)+1:end) = 0;  
BBB_all = sum(exp(SS / tau), 2) - 1;
offsets = n * (1:num_views-1); 
values = arrayfun(@(k) diag(SS, offsets(k)), 1:num_views-1, 'UniformOutput', false);
total_sum = sum(cellfun(@sum, values));
losslcc=sum(log(BBB_all))/(n*num_views)-2*total_sum/ (tau * n * num_views * (num_views-1));
obj_val = losslcc;
obj_history(iter) = obj_val;
    if iter > 1 && abs(obj_history(iter) - obj_history(iter-1)) < tolerance
        break;
    end
end
Yv=cell(num_views, 1);
for v = 1:num_views
Yv{v}=uall{v}./sum(uall{v}, 2);
end
Y = mean(cat(3, Yv{:}), 3);
end