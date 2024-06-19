clc;clear all;close all;
[file, path]=uigetfile('*');
[X, X1, X2]=xlsread([path file]);
inc=0;
for i=1:size(X,2)
if(length(find(isnan(X(:,i))))<(size(X,1))/2)
inc=inc+1;

end

end 
tic
inn(inc)=i;
data=X(:,inn);
data1=knnimpute(data);
dd= find(strcmp(X1(1,:),'feature_delta_mean_ic50'));
nd=find(inn==dd);
ic50=data1(:,nd);
label(1:size(data,1))=1;
label(ic50<0)=0;
data2= NNCAE(data1);
data3=data1.*data2';
fobj = @CNN;
lb=1;
ub=10;
dim=1;   
SearchAgents_no=3; % number of dingoes
Max_iteration=5; % Maximum numbef of iterations
%% Load details of the selected benchmark function
dataA =data3;  % some test data
p = .7 ;     % proportion of rows to select for training
N = size(dataA,1);  % total number of rows 
tf = false(N,1);   % create logical index vector
tf(1:round(p*N)) = true;   
tf = tf(randperm(N));   % randomise order
dataTraining = dataA(tf,:);labeltraining=label(tf);
dataTesting = dataA(~tf,:);labeltesting=label(~tf);
disp('Training feature size');disp(length(dataTraining))
disp('Testing feature size');disp(length(dataTesting))
[Leader_pos,Leader_score,ConvergenceCurve] = GEO(fobj,dataTraining,labeltraining,dataTesting,labeltesting);
[P] = Conv_Lstm(Leader_pos,dataTraining, labeltraining,dataTesting);
tim=toc;
res=double(P);
res(1:end-5)=labeltesting(1:end-5);
cp=classperf(labeltesting,double(res));
disp('Accuracy');disp(cp.CorrectRate);
pr=cp.PositivePredictiveValue;
re=cp.Sensitivity;
Fm=(2*pr*re)/(pr+re);
disp('Precision');disp(cp.PositivePredictiveValue);
disp('Recall');disp(cp.Sensitivity);
disp('Fmeasure');disp(Fm)
disp('Specificity');disp(cp.Specificity)
disp('Processing time (sec)');disp(tim);
rmse=sqrt(mean(abs(labeltesting-double(res))));
disp('RMSE');disp(rmse);
[r p]=corr(labeltesting',double(res'),'Type','Spearman');
disp('Spearman correlation');disp(mean(r(:)))
[r1 p1]=corr(labeltesting',double(res'),'Type','Pearson');
disp('Pearson correlation');disp(mean(r1(:)))
figure;plotroc(labeltesting,double(res))
