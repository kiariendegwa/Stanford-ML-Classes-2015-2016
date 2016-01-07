function [J, grad] = nnCostFunction(nn_params, ...
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

% Feed forward algorithm
X = [ones(m, 1) X];
a_1 = X;
z_2 = a_1*Theta1';
a_2 = sigmoid(z_2);
z_3 = [ones(m, 1) a_2]*Theta2';
h_theta= sigmoid(z_3);

%Remove bias nodes from both Theta1 and Theta2
theta1wtbias = Theta1(:, 2:end);
theta2wtbias = Theta2(:, 2:end);

%Convert Y into a 5000*10 boolean matrix
Y = zeros(size(y, 1), num_labels);
for i=1:m
    y_k = zeros(num_labels, 1);
    y_k(y(i)) = 1;
    Y(i, :) = y_k;
end

cost = -Y.*log(h_theta)-(1-Y).*log(1-h_theta);
J = J+(1/m)*sum(sum(cost))+(lambda/(2*m))*(sum(sum(theta1wtbias.^2))+ sum(sum(theta2wtbias.^2)));

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

delta1 = zeros(size(Theta1));
delta2 = zeros(size(Theta2));

for i = 1:m
    a1_d = a_1(i, :);
    a2_d = a_2(i, :);
    ht = h_theta(i, :);
    yt = Y(i, :);
    
    d3 = (ht - yt)';
 
    z_l2 = [1, z_2(i, :)];
    d2 = Theta2'*d3.*sigmoidGradient(z_l2)';
 
    %Remove delta_2(0)
    d2 = d2(2:end);
    delta1 = delta1 + d2*a1_d;
    %Add bias node to a2 activation layer
    delta2 = delta2 + d3*([1, a2_d]);
    
end

Theta1_grad = Theta1_grad + (1/m)*delta1;
Theta2_grad = Theta2_grad+(1/m)*delta2;

%  Generate the required y value format
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.

Theta1zeroBias = [zeros(size(Theta1, 1), 1) theta1wtbias];
Theta2zeroBias = [zeros(size(Theta2, 1), 1) theta2wtbias];

% -------------------------------------------------------------
Theta1_grad = Theta1_grad + (lambda/m)*Theta1zeroBias;
Theta2_grad = Theta2_grad +(lambda/m)*Theta2zeroBias;
% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
