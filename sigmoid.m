function g = sigmoid(z)
% SIGMOID Calculates the sigmoid function on each element of a matrix.

% 	g = SIGMOID(X) calculates 1/(1+exp(-x)) for each element of the matrix X
%	and returns it as g.
g = 1./(1 + exp(-z));

end


