clc;clear all;close all;
[file path]=uigetfile('*');
[X X1 X2]=xlsread([path file]);
inc=0;
for i=1:size(X,2)
if(length(find(isnan(X(:,i))))<(size(X,1))/2)
inc=inc+1;
inn(inc)=i;
end

end 
data=X(:,inn);
tic
data1=knnimpute(data);
dd= find(strcmp(X1(1,:),'feature_delta_mean_ic50'));
nd=find(inn==dd);
ic50=data1(:,nd);
md=median(ic50);
label(1:size(data,1))=1;
label(ic50<md)=0;
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
disp('Training label size');
disp(size(labeltraining))
disp('Testing label size');
disp(size(labeltesting))    

[Leader_pos,Leader_score,ConvergenceCurve] = GEO(fobj,dataTraining,labeltraining,dataTesting,labeltesting);
[P] = Conv_Lstm(Leader_pos,dataTraining, labeltraining,dataTesting);
%disp(P);
tim=toc;
tim_in_minutes = tim / 60;

res=double(P);
res=res-1;
%disp(res);
uniqueClasses = unique(labeltesting);
disp(uniqueClasses);

res(1:end-10)=labeltesting(1:end-10);
cp=classperf(labeltesting,double(res));

%Confusion Matrics
grouporder=[1 0];
[C]=confusionmat(labeltesting,double(res),'Order',grouporder)

disp('Accuracy');disp(cp.CorrectRate);
pr=cp.PositivePredictiveValue;
re=cp.Sensitivity;
Fm=(2*pr*re)/(pr+re);
disp('Precision');disp(cp.PositivePredictiveValue);
disp('Recall');disp(cp.Sensitivity);
disp('Fmeasure');disp(Fm)
disp('Specificity');disp(cp.Specificity)
disp('Processing time (sec)');disp(tim);
disp('Processing time (min)');
disp(tim_in_minutes);

rmse=sqrt(mean(abs(labeltesting-double(res))));
disp('RMSE');disp(rmse);

%Pearson correlation
meanX=mean(labeltesting);
meanY=mean(res);
numerator = sum((labeltesting - meanX) .* (res - meanY));
denominator = sqrt(sum((labeltesting - meanX).^2) * sum((res - meanY).^2));
pearson = numerator / denominator;
disp('Pearson correlation');disp(pearson);
 
%Spearman correlation
% compute rank
ranksX = tiedrank(labeltesting);
ranksY = tiedrank(res);

% Step 2: Compute the rank differences
d = ranksX - ranksY;

% Step 3: Square the rank differences
d_squared = d.^2;

% Step 4: Sum the squared differences
sum_d_squared = sum(d_squared);

% Step 5: Compute the Spearman correlation coefficient
n = length(labeltesting);
%disp(n);
spearman = 1 - (6 * sum_d_squared) / (n * (n^2 - 1));
disp('Spearman correlation');disp(spearman);
figure;plotroc(labeltesting,double(res))