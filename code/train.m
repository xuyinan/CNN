function [model, loss] = train(model,input,label,params,numIters)

% Initialize training parameters
% This code sets default values in case the parameters are not passed in.

% Learning rate
if isfield(params,'lr') lr = params.lr;
else lr = .01; end
% Weight decay
if isfield(params,'wd') wd = params.wd;
else wd = .0005; end
% Batch size
if isfield(params,'batch_size') batch_size = params.batch_size;
else batch_size = 128; end

% There is a good chance you will want to save your network model during/after
% training. It is up to you where you save and how often you choose to back up
% your model. By default the code saves the model in 'model.mat'
% To save the model use: save(save_file,'model');
if isfield(params,'save_file') save_file = params.save_file;
else save_file = 'model.mat'; end

% update_params will be passed to your update_weights function.
% This allows flexibility in case you want to implement extra features like momentum.
update_params = struct('learning_rate',lr,'weight_decay',wd);


% define parameters for training model
num_layers = numel(model.layers);
loss_scaler = 100;
maxloss = 5;
loss = zeros(1,numIters);
iter = 1;
backprop = true;
while loss_scaler > maxloss && iter < numIters
	% TODO: Training code
    if iter > 1
        model = updated_model;
    end
    
    [output_final,activations] = inference(model,input);
    hyper_params = size(output_final);
    
    [loss_scaler, dv_input] = loss_crossentropy(output_final, label, hyper_params, backprop);
    grad = calc_gradient(model, input, activations, dv_input);
    updated_model = update_weights(model,grad,update_params);
    loss(iter) = loss_scaler;
    iter = iter+1;
end
loss = loss(loss>0);

fprintf('iteration: %i...',iter);

save(save_file,'model')
