function feat1= NNCAE(data)
[N, n] = size(data);
X = data;

% Define the autoencoder architecture
inputSize = size(data, 2); % Input size
hiddenSize = 128; % Number of units in the hidden layer

% Create and train the autoencoder
autoencoder = feedforwardnet(hiddenSize);
autoencoder.layers{1}.transferFcn = 'poslin'; % Non-negative activation function
autoencoder.layers{2}.transferFcn = 'poslin'; % Non-negative activation function


% Set training options
options = trainingOptions('adam', 'MaxEpochs', 100, 'MiniBatchSize', 32);

% Train the autoencoder
autoenc = trainAutoencoder(X',1,'MaxEpochs',400,...
'DecoderTransferFunction','purelin');
feat1 = encode(autoenc,X');
