function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C_set = [0.01,0.03,0.1,0.3,1,3,10,30];
sigma_set = [0.01,0.03,0.1,0.3,1,3,10,30];

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
results = [];
for C_index = 1:length(C_set)
  for sigma_index = 1:length(sigma_set)
    C_current = C_set(C_index);
    sigma_current = sigma_set(sigma_index);
    model= svmTrain(X, y, C_current, @(x1, x2) gaussianKernel(x1, x2, sigma_current));
    pred = svmPredict(model, Xval);
    error = mean(double(pred ~= yval));
    current_result = [C_current sigma_current error];
    results = [results ; current_result];
  end;
end;

[min_err index] = min(results(:,3));

C = results(index,1);
sigma = results(index,2);


% =========================================================================

end
