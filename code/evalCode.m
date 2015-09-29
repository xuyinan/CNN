%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%% This is the evaluation for part two %%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
addpath layers;
% clear;close all;
%% load dataset
load_MNIST_data;
assert(MNIST_loaded == true);
%% define parameters
% subset of dataset
% The dataset is including four parts: 
train_batch_size = 200;
test_batch_size = 128;

% training switch
train_model_bool = false;   % true; false
% new model switch
new_model = false;   % true; false
% model size switch
model_size_disp = false;   % true; false

% training parameters
train_params = struct('lr',0.001,'wd',0.0005,'batch_size',train_batch_size);
% time of training iteration
train_times = 50;
train_iteration = 1000;

%% training the model
if train_model_bool
    
    % model---96.21%
    layers = [init_layer('conv',struct('filter_size',5,'filter_depth',1,'num_filters',8)) ...
        init_layer('pool',struct('filter_size',5,'stride',4)) ...
        init_layer('flatten',struct('num_dims',4)) ... 
        init_layer('linear',struct('num_in',25*8,'num_out',10)) ...
        init_layer('softmax',[])];


    %%
    if new_model
        % initial network layers and output network model
        fprintf('Initializing Model...');tic;
        model = model_initialization(layers, [28 28 1 train_batch_size], [10 train_batch_size], model_size_disp);
        t_ini_model = toc; fprintf('Done. Time: %1.2f sec.\n', t_ini_model);
        fprintf('-----------------------------------------------------\n');
    else
        fprintf('Loading Model for training...');tic;
        model = load('model.mat');
        if isfield(model,'model')
            model = model.model;
        end
        t_load = toc; fprintf('Done. Time: %1.2f sec.\n', t_load);
        fprintf('-----------------------------------------------------\n');
    end
    % training the model
    acc = 0;
    i = 1;
    accuracy = zeros(1,train_times);
    while i <= train_times && acc<0.97
        train_sub = randi(size(train_data,4), [1 train_batch_size]);
        train_data_subset = train_data(:,:,:,train_sub);
        train_label_subset = train_label(train_sub);
        
%         test_sub = randi(size(test_data,4), [1 test_batch_size]);
%         test_data_subset = test_data(:,:,:,test_sub);
%         test_label_subset = test_label(test_sub);
        
        test_data_subset = test_data;
        test_label_subset = test_label;
        
        % training the model
        fprintf('\nTraining Model times %d...', i); tic;
        [model, train_loss] = train(model, train_data_subset, train_label_subset, train_params, train_iteration);
        t_train = toc; fprintf('Done. Time: %1.2f sec.\n', t_train);
        figure;
        plot(train_loss)
        % evaluate the test dataset 
        fprintf('-----Evaluating the accuracy of the model...');tic;
        [loss, acc] = evalAccPerct(model,test_data_subset,test_label_subset);
        accuracy(i) = acc;
        t_eval_test = toc; fprintf('Done. Time: %1.2f sec.\n', t_eval_test);
        fprintf('-----Accuracy percet: %1.2f%%. Loss: %1.2f.\n', 100*acc, loss);
        i = i+1;
    end
else
    fprintf('  Loading Model for Evaluating...');tic;
    model = load('model.mat');
    if isfield(model,'model')
        model = model.model;
    end
    t_load = toc; fprintf('Done. Time: %1.2f sec.\n', t_load);
    fprintf('-----------------------------------------------------\n');
    % evaluate the test dataset
    fprintf('-----Evaluating the accuracy of the model...');tic;
    [loss, acc] = evalAccPerct(model,test_data,test_label);
    t_eval_test = toc; fprintf('Done. Time: %1.2f sec.\n', t_eval_test);
    fprintf('-----Accuracy percet: %1.2f%%. Loss: %1.2f.\n', 100*acc, loss);
end
    
    
% 
% clear;



