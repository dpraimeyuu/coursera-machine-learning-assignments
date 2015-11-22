function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
k = size(theta)(1)
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %

    thetaBefore = theta;
    for thetaIndex = 1:k
      dTheta = zeros(k, 1);
      for i=1:m
        dTheta(thetaIndex) = dTheta(thetaIndex) + (thetaBefore'*X(i,:)' - y(i))*X(i,thetaIndex);
      end
      theta(thetaIndex) = theta(thetaIndex) - (alpha/m)*dTheta(thetaIndex);
    end




    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
