% input: batch_size * num_in_nodes 
% output: output of last layer
% activations: outputs of all layers

function [output,activations] = inference(model,input)
% Do forward propagation through the network to get the activation
% at each layer, and the final output

num_layers = numel(model.layers);
activations = cell(num_layers,1);

% TODO: FORWARD PROPAGATION CODE
backprop = false;
dv_output = zeros(size(input));

for i = 1:num_layers
    if i > 1
        input = activations{i-1};
    end
    [activations{i}, ~, ~] = model.layers(i).fwd_fn(input, model.layers(i).params, model.layers(i).hyper_params, backprop, dv_output);
end

output = activations{end};


end