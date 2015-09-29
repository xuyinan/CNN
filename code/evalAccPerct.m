function [loss, acc] = evalAccPerct(model,test_data,test_label)

[test_output,~] = inference(model,test_data);
hyper_params = size(test_output);
[loss, ~] = loss_crossentropy(test_output, test_label, hyper_params, false);

assert(size(test_label,1) == size(test_output,2));
[~,test_result] = max(test_output);
% difference between the outputs and the labels
differefce = test_label - test_result';
% evaluate the percent of accuracy
eval_count = differefce(differefce == 0);
acc = length(eval_count)/length(test_label);


end


