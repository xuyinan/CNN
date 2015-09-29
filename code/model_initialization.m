function model = model_initialization(layers, input_size, output_size, disp)
% Basic script to create a new network model
if nargin < 1
    input_size = [28 28 1 128];
    output_size = [10 128];
end
if nargin < 3
    disp = true;
%     disp = false;
end

addpath layers;

% Initialize a network model given an array of layers.
% Expected input and output size must be provided to verify that the network
% is properly defined.
model = struct('layers',layers,'input_size',input_size,'output_size',output_size);

if disp
    % Check that layer input/output sizes are correct
    % Batch sizes of 1 and greater than 1 are used to ensure that both cases are handled properly by the code
    num_layers = length(model.layers);
    input = rand(input_size);
    
    % Run inference to get intermediate activation sizes, and final output size
    [output,act] = inference(model,input);
    network_output_size = size(output);
    
    % While designing your model architecture it can be helpful to know the
    % intermediate sizes of activation matrices passed between layers. 'display'
    % is an option you set when you call 'init_model'.
    
	display('Input size:');
	display(input_size);
	for i = 1:num_layers-1
		fprintf('Layer %d output size:\n',i);
		outpurSize = size(act{i});
        display(outpurSize);
	end
	display('Final output size:');
	display(network_output_size);
	display('Provided output size:');
	display(output_size);
end

% If you defined all of your layers correctly you should know the final
% size of the output matrix, this is just a sanity check.

% assert(isequal(network_output_size,output_size),...
% 		'Network output does not match up with provided output size');

end


