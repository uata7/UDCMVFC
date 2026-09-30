function [Y1, obj_history] = UDCMVFC(X, c,max_iters,tolerance,r,a,tau,l,beta)

rng(4, 'twister');  
num_views = length(X); 
n = size(X{1}, 1);

X_concat = [X{:}];
[label1, ~] = litekmeans(X_concat, c, 'MaxIter', 100, 'Replicates', 10);
n1 = length(label1);
Y = full(sparse(1:n1, label1, 1, n1, c));

M = cell(num_views, 1);
YTT=(Y.^r)'; 
YTTT=sum(YTT,2);
momen = cell(num_views, 1);
 
d = zeros(num_views, 1);  
for v = 1:num_views
M{v}= YTT*X{v} ./ YTTT;
momen{v} = zeros(size(M{v}));
d(v)=size(M{v}, 2);   
end
obj_history = zeros(max_iters, 1);
uall = cell(num_views, 1);
normall = cell(num_views, 1); 

for  v = 1:num_views
     dtt = pdist2(X{v}, M{v},'euclidean');
     uall{v} = dtt.^(2/(1-r));
     normall{v} = sqrt(sum(uall{v}.^2, 2)); 
end

for iter = 1:max_iters
    uuualls = vertcat(uall{:});     
    nnnorms = vertcat(normall{:});   
    S = uuualls * uuualls'; 
    normpp = nnnorms * nnnorms';
    S = S ./ normpp;      
    S(1:size(S,1)+1:end) = 0;  

    expS = exp(S / tau);
    Bs = sum(expS, 2) - 1;

    [dlcc_dm_all] = UDCMVFC_grad(X,M,uall, uuualls,nnnorms,S,r,tau,l,expS,Bs,d);
    offsets = n * (1:num_views-1); 
    values = arrayfun(@(k) diag(S, offsets(k)), 1:num_views-1, 'UniformOutput', false); 
    total_sum = sum(cellfun(@sum, values));
    losslcc=sum(log(Bs))/(n*num_views)-2*total_sum/ (tau * n * num_views * (num_views-1));
    obj_history(iter) = losslcc;  
    if iter > 1 && abs(obj_history(iter) - obj_history(iter-1)) < tolerance
        break;
    end

    for v0 = 1:num_views  
        momen{v0} = beta * momen{v0} + dlcc_dm_all{v0}; 
        M{v0} = M{v0} - a * momen{v0};
    end
    for v = 1:num_views
        uall{v}=pdist2(X{v}, M{v},'euclidean').^(2/(1-r));  
        normall{v} = sqrt(sum(uall{v}.^2, 2));    
    end
end

Yv=cell(num_views, 1);
for v = 1:num_views
    Yv{v}=uall{v}./sum(uall{v}, 2);
end

Y1 = mean(cat(3, Yv{:}), 3);
end


function  [dlcc_dm_all] =  UDCMVFC_grad(views,M,uall, uuualls,nnnorms,S,r,tau,l,expS,Bs,d)
num_views = length(uall); 
n = size(uall{1}, 1); 
dlcc_dm_all = cell(num_views, 1); %
%--
P = kron(ones(num_views),eye(n));
P(1:(n * num_views)+1:end) = 0;
Q = uuualls ./ nnnorms;
part1 = (P' * Q) ./ nnnorms; 
coef1 =sum(P .* S,2) ./ (nnnorms.^2);
part2 = coef1 .* uuualls;  
result1 = part1 - part2;
%--
C = bsxfun(@plus, 1./Bs, 1./Bs').* expS;      
C(1:size(C,1)+1:end) = 0; 
result2 = C*Q./nnnorms - (sum(C.*S,2)./(nnnorms.^2) ).*uuualls; 
result =(result2 -(2/(num_views-1))*result1)/tau; 
allview_scale = repelem(d.^l, n);
dLcc_du_nvc = allview_scale .* result * (1/(num_views*n));
W_all = dLcc_du_nvc.* (uuualls.^r);

for v0 = 1:num_views
    rows = (v0-1)*n + (1:n);
    W = W_all(rows, :);              
    row = sum(W, 1);                 
    dlcc_dm = (row') .* M{v0} - W' * views{v0};
    dlcc_dm_all{v0} = 2/(1-r) * dlcc_dm;
end
end


