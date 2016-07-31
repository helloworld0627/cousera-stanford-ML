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
                 num_labels, (hidden_layer_size + 1));

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

% convert y values to matrix (each value presented as vector)
for i = 1:m
  label_vector(i, y(i)) = 1;
end

% add '1' for bias term
X = [ones(m,1) X];

% compute z2 and a2
z2 = (X * Theta1');
a2 = sigmoid(z2);
a2 = [ones(m, 1) a2];

% compute z3 and a3
z3 = a2 * Theta2';
a3 = sigmoid(z3);

% compute cost
for i = 1:num_labels
  % label value presented as column(i)
  col_vector = label_vector(:, i);
  col_a3 = a3(:, i);

  % apply cost function
  j = sum(-col_vector .* log(col_a3) - (1 - col_vector) .* log(1 - col_a3));
  J += j;
end

% non Reg cost
J = J / m;

% compute regularized cost function
regTheta1 = [Theta1(:, 2:end)];
regTheta2 = [Theta2(:, 2:end)]; 
squareRegTheta1 = regTheta1.^2;
squareRegTheta2 = regTheta2.^2;
sumOfSquareRegTheta1 = sum(squareRegTheta1(:));
sumOfSquareRegTheta2 = sum(squareRegTheta2(:));
sumOfSquareRegTheta = sumOfSquareRegTheta1 + sumOfSquareRegTheta2;
regCost = lambda * sumOfSquareRegTheta / (2*m);

% Reg cost
J += regCost;
  
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

for t=1:m
  % already append one in part 1 
  a1 = X(t, :);
  z2 = a1 * Theta1';
  a2 = sigmoid(z2);
  a2 = [1 a2];

  z3 = a2 * Theta2';
  a3 = sigmoid(z3);

  d3 = a3 - label_vector(t, :);

  z2 = [1 z2];
  d2 = d3*Theta2 .* sigmoidGradient(z2);
  d2 = d2(2:end);

  Theta2_grad = Theta2_grad + d3' * a2;
  Theta1_grad = Theta1_grad + d2' * a1; 

end
  
Theta1_grad = Theta1_grad / m;
Theta2_grad = Theta2_grad / m;


% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + (lambda / m) .* Theta1(:,2:end);
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + (lambda / m) .* Theta2(:,2:end);













% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
