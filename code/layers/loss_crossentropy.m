% ======================================================================
% Matrix size reference:
% ----------------------------------------------------------------------
% input: num_nodes * batch_size
% labels: batch_size * 1
% ======================================================================

function [loss, dv_input] = loss_crossentropy(input, labels, hyper_params, backprop)

assert(max(labels) <= size(input,1));

% TODO: CALCULATE LOSS
loss = 0;
log_input = log(input);
dv_input_out = zeros(size(input));
for i = 1:size(input,2)
    loss = loss - log_input(labels(i),i);
    dv_input_out(labels(i),i) = -1/input(labels(i),i);
end

dv_input = zeros(size(input));
if backprop
	% TODO: BACKPROP CODE
    dv_input = dv_input_out;
end


end
