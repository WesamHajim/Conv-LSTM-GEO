function [P] = Conv_Lstm(fz,data_x, target_y,data_test)

% Input paramters (arguments) are:
%   data_x: 4-D double containing the input data in the following form
%       [128 1 1 numberOfReadings]
%   target_y: 1-D Categorial containing the signal labels
%       [numberOfReadings 1]

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fz;
[m, n]=size(data_x);
[mm, nn]=size(data_test);
inputdata=reshape(data_x,[n m]);
inp=reshape(data_test,[nn mm]);

target=categorical(target_y);
size(inputdata)
size(target)
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
P = classify(net,inp, ...
    'MiniBatchSize',miniBatchSize, ...
    'SequenceLength','longest');














