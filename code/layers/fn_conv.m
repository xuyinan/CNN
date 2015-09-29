% ======================================================================
% Matrix size reference:
% input: in_height * in_width * num_channels * batch_size
% output: out_height * out_width * num_filters * batch_size
% hyper parameters: (used for options like stride, padding (neither is required for this project))
% params.W: filter_height * filter_width * filter_depth * num_filters
% params.b: num_filters * 1
% dv_output: same as output
% dv_input: same as input
% grad.W: same as params.W
% grad.b: same as params.b
% ======================================================================

function [output, dv_input, grad] = fn_conv(input, params, hyper_params, backprop, dv_output)

[~,~,num_channels,batch_size] = size(input);
[~,~,filter_depth,num_filters] = size(params.W);
assert(filter_depth == num_channels, 'Filter depth does not match number of input channels');

out_height = size(input,1) - size(params.W,1) + 1;
out_width = size(input,2) - size(params.W,2) + 1;
output = zeros(out_height,out_width,num_filters,batch_size);
% TODO: FORWARD CODE
for out_j = 1:batch_size
    for out_i = 1:num_filters
        for out_m = 1:num_channels
            output(:,:,out_i,out_j) = output(:,:,out_i,out_j) + conv2(input(:,:,out_m,out_j), params.W(:,:,out_m,out_i), 'valid');
        end
        output(:,:,out_i,out_j) = output(:,:,out_i,out_j) + params.b(out_i);
    end
end

dv_input = [];
grad = struct('W',[],'b',[]);

if backprop
	dv_input = zeros(size(input));
	grad.W = zeros(size(params.W));
	grad.b = zeros(size(params.b));
	% TODO: BACKPROP CODE
        
    %% dv_input
    for dv_j = 1:batch_size
        for dv_i = 1:filter_depth 
            for dv_m = 1:num_filters
                dv_input(:,:,dv_i,dv_j) = dv_input(:,:,dv_i,dv_j) + conv2(dv_output(:,:,dv_m,dv_j), rot90(params.W(:,:,dv_i,dv_m),2), 'full');
            end
        end
    end
    
    %% gradient of W
    for W_j = 1:num_filters
        for W_i = 1:num_channels
            for W_m = 1:batch_size
                grad.W(:,:,W_i,W_j) = grad.W(:,:,W_i,W_j) + conv2(rot90(input(:,:,W_i,W_m),2), dv_output(:,:,W_j,W_m), 'valid');
            end
        end
    end    
    
    %% gradient of b
    for b_i = 1:size(dv_output,3)
        grad.b(b_i) = sum(sum(sum(dv_output(:,:,b_i,:))));
    end
    
end


end
