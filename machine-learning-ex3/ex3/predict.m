function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% n = n1
% size(Theta1) = n2 x (n1+1)
% size(Theta2) = n3 x (n2+1)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%


X = [ones(m, 1), X]; %m x n1+1  5000 x 401


Z2 = X * Theta1'; %m x n2   5000 x 25
A2 = sigmoid(Z2); %m x n2   5000 x 25
A2 = [ones(m, 1) A2];  %m x (n2+1)  5000 x 26

Z3 = A2 * Theta2';  %m x n3  5000 x 10
A3 = sigmoid(Z3);  %m x n3 5000 x 10 

[foo c] = max(A3');
p = c';

% =========================================================================


end
