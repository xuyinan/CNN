function updated_model = update_weights(model,grad,hyper_params)

num_layers = length(grad);
lr = hyper_params.learning_rate;
wd = hyper_params.weight_decay;

% TODO: Update the weights of each layer in your model based on the calculated gradients
for i = 1:num_layers
    W = model.layers(i).params.W;
    b = model.layers(i).params.b;
    dW = grad{i}.W;
    db = grad{i}.b;
    model.layers(i).params.W = W - lr*dW - wd*W;
    model.layers(i).params.b = b - lr*db - wd*b;
end

updated_model = model;

end