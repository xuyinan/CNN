% ======================================================================
% Matrix size reference:
% ----------------------------------------------------------------------
% input: num_classes * batch_size
% output: num_classes * batch_size
% ======================================================================

function [output, dv_input, grad] = fn_softmax(input, params, hyper_params, backprop, dv_output)

[num_classes,batch_size] = size(input);

% output = zeros(num_classes, batch_size);
% TODO: FORWARD CODE
output = exp(input)./repmat(sum(exp(input),1), [num_classes,1]);

dv_input = [];

% This is included to maintain consistency in the return values of layers,
% but there is no gradient to calculate in the softmax layer since there
% are no weights to update.
grad = struct('W',[],'b',[]); 

if backprop
    dv_input = zeros(size(input));
	% TODO: BACKPROP CODE
    delta = eye(num_classes);
    for k = 1:batch_size
        output_expand = repmat(output(:,k),[1 num_classes]);
        dy_dx = output_expand'.*(delta- output_expand);
        dv_input(:,k) = dy_dx * dv_output(:,k);
    end
end

end
