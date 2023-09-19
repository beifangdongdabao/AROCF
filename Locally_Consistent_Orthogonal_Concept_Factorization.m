clc
clear all
close all

addpath(genpath('./utils'))
dsPath = './data/';

ds = {'Cora_ML_uni','Cora_OS_uni','20news_uni','text1_uni','TDT2_10_uni','SearchSnippets-lite','StackOverflow-lite'};
dataName = 'text1_uni'
load(strcat(dsPath,dataName));

X = (mapstd(X))';   % mpstd(X)应保证X为n*d 

RESULT = [];
%% 超参数
m = 10;
lmd = 1;

for iter=1:50
     iter
     tic 
    % 初始化变量
    [d,N] = size(X);
    NC = length(unique(Y));
    [~,landmark] = litekmeans(X',m);
    [z, ~, ~, ~, ~] = ConstructA_NP(X, landmark',512);
    Z = z*z';
    D = diag(sum(Z,2));
    L = D - Z;
    OBJ = [];  
    % Our method
    H = initializeG(N,NC);
    U = initialize(N,NC);

    for i=1:10
        % update H
        a = max(max(L)) * lmd;
        A = (ceil(a)+1) * eye(size(L)) - lmd * L;
        B = X' * X * U;
        C = 2 * B + 2 * A * H;
        [UU,TT,WW] = svd(C,'econ');
        H = UU * WW;
        % update U
        U = H;
        obj = norm(X - X * U * H','fro')^2 + lmd * trace(H'*L*H);
        OBJ = [OBJ,obj];
    end
    t = toc;
    [maxv,ind]=max(H,[],2);
    [result] = ClusteringMeasure(Y, ind)
    RESULT=[RESULT;result,t];
 end
record=[
    mean(RESULT(:,1)),std(RESULT(:,1));
     mean(RESULT(:,2)),std(RESULT(:,2));
     mean(RESULT(:,3)),std(RESULT(:,3));
     mean(RESULT(:,4)),std(RESULT(:,4));
     mean(RESULT(:,5)),std(RESULT(:,5));];

S_path = [char(dataName),'.txt'];
dlmwrite(S_path,record','-append','delimiter','\t','newline','pc');
      






            
        


