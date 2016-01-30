function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 0.01;
sigma = 0.01;

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

min_prediction_error = 1000;
C_opt = 0;
sigma_opt = 0;

while (C <= 30)
  sigma = 0.01;
  while (sigma <= 30)
    model = svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma), 1e-3, 20);
    predictions = svmPredict(model, Xval);
    prediction_err = mean(double(predictions ~= yval))
    if prediction_err < min_prediction_error
      sigma_opt = sigma;
      C_opt = C;
      min_prediction_error = prediction_err;
    endif
    sigma = sigma * 3;
  endwhile
  C = C * 3
endwhile

C = C_opt;
sigma = sigma_opt;

% =========================================================================

end
