function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;
v= [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30]
m=size(X,1)
r=size(Xval,1)
% ====================== YOUR CODE HERE ======================
% Instructions. Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note. You can compute the prediction error using 
%        mean(double(predictions ~= yval))

%
min = 1
count=0
for i=1:length(v)
  for j=1:length(v)
    model= svmTrain(X, y, v(i), @(x1, x2) gaussianKernel(x1, x2, v(j))); %training with train set
    predictions = svmPredict(model,Xval) % testing on CV Set
    count+=1
    fprintf("Errors\t\tC\t\tsigma\t.......................................%d\n",count)
    fprintf("%f\t%f\t%f\t\n",mean(double(predictions ~= yval)),v(i),v(j))
    if(min>mean(double(predictions ~= yval)))
      C= v(i)
      sigma = v(j)
      min = mean(double(predictions ~= yval))
    end
  endfor
endfor








% =========================================================================

end
