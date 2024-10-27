function [P] = Conv_Lstm(fz,data_x, target_y,data_test)

% Input paramters (arguments) are:
%   data_x: 4-D double containing the input data in the following form
%       [128 1 1 numberOfReadings]
%   target_y: 1-D Categorial containing the signal labels
%       [numberOfReadings 1]

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fz
[m n]=size(data_x);
[mm nn]=size(data_test);
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
maxEpochs = 300;
miniBatchSize = 120;

options = trainingOptions('adam', ...
    'ExecutionEnvironment','cpu', ...
    'InitialLearnRate',0.9,...
    'GradientThreshold',0.5, ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'SequenceLength','longest', ...
    'Shuffle','never', ...
    'Plots', 'training-progress',...
    'Verbose',0);
net = trainNetwork(inputdata,target,layers,options);
miniBatchSize = 120;
P = classify(net,inp, ...
    'MiniBatchSize',miniBatchSize, ...
    'SequenceLength','longest');
%disp(P);
end














