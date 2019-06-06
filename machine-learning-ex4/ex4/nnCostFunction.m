function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1)); % num_labels x (hidden_layer_size + 1)

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

lsum = 0;
for i=1:m  
  a1 = X(i, :)';  % input_layer_size x 1
  a1b = [1; a1]; % (input_layer_size +1 ) x 1
  %Theta1: hidden_layer_size, (input_layer_size + 1)
  z2 = Theta1 * a1b; %hidden_layer_size x 1
  a2 = sigmoid(z2); %hidden_layer_size x 1
  
  a2b = [1; a2];  %(hidden_layer_size+1) x 1
  %Theta2: num_labels x (hidden_layer_size + 1)
  z3 = Theta2 * a2b; %num_labels x 1
  a3 = sigmoid(z3); %num_labels x 1
  h = a3; %num_labels x 1
  
  bin_y = zeros(num_labels, 1);
  bin_y(y(i)) = 1;
  for k=1:num_labels
    lsum += -bin_y(k)*log(h(k)) - (1-bin_y(k)) * log(1 - h(k));
  end

    
end

J = (1/m)*lsum;

%Theta1_reg = Theta1(:, 2:(input_layer_size + 1));
%Theta2_reg = Theta2(:, 2:(hidden_layer_size + 1));
Theta1_reg = Theta1(:, 2:end);
Theta2_reg = Theta2(:, 2:end);


reg = ( sum(sum(Theta1_reg.^2))+sum(sum(Theta2_reg.^2))  )  * lambda / (2*m);

J += reg;



% -------------------------------------------------------------

% =========================================================================

%Theta2_Delta = zeros(num_labels, hidden_layer_size);
%Theta1_Delta = zeros(hidden_layer_size, input_layer_size);

Theta2_Delta = zeros(size(Theta2));  %num_labels x (hidden_layer_size + 1)
Theta1_Delta = zeros(size(Theta1));  %hidden_layer_size, (input_layer_size + 1)

for i=1:m  
  a1 = X(i, :)';  % input_layer_size x 1
  a1b = [1; a1]; % (input_layer_size +1 ) x 1
  %Theta1: hidden_layer_size, (input_layer_size + 1)
  z2 = Theta1 * a1b; %hidden_layer_size x 1
  a2 = sigmoid(z2); %hidden_layer_size x 1
  
  a2b = [1; a2];  %(hidden_layer_size+1) x 1
  %Theta2: num_labels x (hidden_layer_size + 1)
  z3 = Theta2 * a2b; %num_labels x 1
  a3 = sigmoid(z3); %num_labels x 1
  h = a3; %num_labels x 1
  
  bin_y = zeros(num_labels, 1);
  bin_y(y(i)) = 1;
  
  delta_l = a3 - bin_y; %num_labels x 1
  delta_hidden = (Theta2' * delta_l)(2:end) .* sigmoidGradient( z2 );  %hidden_layer_size x 1
  % (hidden_layer_size + 1) x num_labels' x %num_labels x 1
  %delta_hidden = (Theta2' * delta_l) .* [0; sigmoidGradient( z2 )];  (%hidden_layer_size + 1) x 1
  
  Theta2_Delta += delta_l * a2b';  %num_labels x 1    1 x (hidden_layer_size + 1) 
  Theta1_Delta += delta_hidden * a1b';  %hidden_layer_size x 1   1 x (input_layer_size+1)
end

%disp('Theta1_Delta size: '); size(Theta1_Delta)
%disp('Theta2_Delta size: '); size(Theta2_Delta)

%Theta1WithZerosColumn = [zeros(size(Theta1, 1), 1)  ]
Theta1WithZerosColumn = Theta1 .* (1:size(Theta1, 2) > 1);
Theta2WithZerosColumn = Theta2 .* (1:size(Theta2, 2) > 1);

Theta1_grad = Theta1_Delta ./ m + lambda/m*Theta1WithZerosColumn;
Theta2_grad = Theta2_Delta ./ m + lambda/m*Theta2WithZerosColumn;

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
