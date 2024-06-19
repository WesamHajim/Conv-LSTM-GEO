function [err] = CNN(fz,data_x, target_y)

% Input paramters (arguments) are:
%   data_x: 4-D double containing the input data in the following form
%       [128 1 1 numberOfReadings]
%   target_y: 1-D Categorial containing the signal labels
%       [numberOfReadings 1]

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fz;
[m, n]=size(data_x);
inputdata=reshape(data_x,[n m]);
target=categorical(target_y);
inputSize = n;
numHiddenUnits = fz;
numClasses = length(unique(target));

layers = [ ...
    sequenceInputLayer(n)
    lstmLayer(numHiddenUnits)
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];
maxEpochs = 100;
miniBatchSize = 27;

options = trainingOptions('adam', ...
    'ExecutionEnvironment','cpu', ...
    'GradientThreshold',1, ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'SequenceLength','longest', ...
    'Shuffle','never', ...
    'Verbose',0);
net = trainNetwork(inputdata,target,layers,options);
miniBatchSize = 27;
P = classify(net,inputdata, ...
    'MiniBatchSize',miniBatchSize, ...
    'SequenceLength','longest');
err=1-(length(P==target)/length(target));








