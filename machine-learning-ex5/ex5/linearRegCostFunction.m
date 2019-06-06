function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

%X (m x n)
%theta n x 1

%X * theta (m x 1)


h = X * theta; %m x 1

reg = 0.5*lambda/m * sum(theta(2:end).^2);

J = 0.5/m*sum((h-y).^2) + reg;

%grad
hy = h-y;  %m x 1

thetaWithFirstZero = [0; theta(2:end)];

grad = 1/m * (X' * hy) + lambda / m * thetaWithFirstZero ; %n x 1

% =========================================================================

grad = grad(:);

end
