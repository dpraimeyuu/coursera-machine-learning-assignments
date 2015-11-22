function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%
m = size(X, 1);
for j=1:m
  sign = "+";
  if(y(j) == 0)
    sign = "o";
  end;
  plotSymbol = strcat("k", sign);
  plot(X(j, 1), X(j, 2), plotSymbol);
end







% =========================================================================



hold off;

end
