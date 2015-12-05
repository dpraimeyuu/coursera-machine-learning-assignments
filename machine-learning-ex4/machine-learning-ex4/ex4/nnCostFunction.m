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
K = size(Theta2, 1);
         
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

X_with_bias = [ones(m, 1) X];
J_regularization = zeros(2, 1);

theta1NodesNumber = size(Theta1, 1);
theta1NodeInputsNumber = size(Theta1, 2);

theta2NodesNumber = size(Theta2, 1);
theta2NodeInputsNumber = size(Theta2, 2);

outputLayerErrors = zeros(theta2NodesNumber, 1);
hiddenLayerErrors = zeros(theta1NodesNumber, 1);
accumulatedOutputLayerErrors = zeros(theta2NodesNumber, theta2NodeInputsNumber);
accumulatedHiddenLayerErrors = zeros(theta1NodesNumber, theta1NodeInputsNumber);
for i = 1:m
  for k = 1:K
    yCoded = zeros(K, 1);
    yCoded(y(i)) = 1;
    [val index] = max(yCoded);
    yVal = 0;
    if index == k
      yVal = 1;
    end
    a2 = zeros(size(Theta2, 2) - 1, 1)';
    a3 = zeros(size(Theta2, 1), 1)';

    a2 = X_with_bias(i, :) * Theta1';
    z2 = a2;
    a2 = sigmoid(a2);
    a2 = [1 a2];
    a3 = a2 * Theta2';
    a3 = sigmoid(a3);
    J = J + ((-yVal * log(a3(k))) - ((1 - yVal) * log(1 - a3(k))));
    
    outputLayerErrors(k) = a3(k) - yVal;
  end
  dSigmoid = (a2 .* (1 - a2));
  thetaTimesErrors = Theta2' * outputLayerErrors;
  hiddenLayerErrors = thetaTimesErrors .* dSigmoid';
  hiddenLayerErrors = hiddenLayerErrors(2:end);
  a1 = X_with_bias;
  accumulatedHiddenLayerErrors = accumulatedHiddenLayerErrors + (hiddenLayerErrors * a1(i, :));
  accumulatedOutputLayerErrors = accumulatedOutputLayerErrors + (outputLayerErrors * a2);
end
accumulatedHiddenLayerErrors = accumulatedHiddenLayerErrors / m;
accumulatedHiddenLayerErrors(:,2:end) = accumulatedHiddenLayerErrors(:,2:end) + (lambda * Theta1(:,2:end))/m;
accumulatedOutputLayerErrors = accumulatedOutputLayerErrors / m;
accumulatedOutputLayerErrors(:,2:end) = accumulatedOutputLayerErrors(:,2:end) + (lambda * Theta2(:,2:end))/m;
p = 1;
for j = 1: theta1NodesNumber
  for k = 2: theta1NodeInputsNumber
    J_regularization(p) = J_regularization(p) + Theta1(j, k)^2;
  end
end

p = 2;
for j = 1: theta2NodesNumber
  for k = 2: theta2NodeInputsNumber
    J_regularization(p) = J_regularization(p) + Theta2(j, k)^2;
  end
end

J = J / m;
J_regularization_final = (lambda * sum(J_regularization)) / (2*m);
J = J + J_regularization_final;

Theta1_grad = accumulatedHiddenLayerErrors;
Theta2_grad = accumulatedOutputLayerErrors;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
