function g = sigmoidGradient(z)
% Compute the gradient of the sigmoid function evaluated at
% each value of z (z can be a matrix, vector or scalar).

g = sigmoid(z).*(1 - sigmoid(z));


end
