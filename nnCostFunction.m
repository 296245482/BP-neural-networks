function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

%forward and calculate the regularized cost function
X = [ones(m,1) X];
for i = 1:m
    yk = zeros(num_labels,1);
    yk(y(i)) = 1;
    a2 = sigmoid(Theta1*X(i,:)');
    a2 = [1;a2];
    a3 = sigmoid(Theta2*a2);
    J = J + sum( -yk.*log(a3) - (1-yk).*log(1-a3));
end

J = J/m + (lambda/(2*m)) * ((sum(sum(Theta1(:,2:end).^2)))+(sum(sum(Theta2(:,2:end).^2))));

%Back Propagation to calculate the Theta gradient
Delta1 = 0;
Delta2 = 0;
for t = 1:m
    yk = zeros(num_labels,1);
    yk(y(t)) = 1;
    a1 = X(t,:)';
    z2 = Theta1 * a1;
    a2 = sigmoid(z2);
    a2 = [1;a2];
    z3 = Theta2 * a2;
    a3 = sigmoid(z3);
    delta3 = a3 - yk;
    delta2 = Theta2' * delta3 .* sigmoidGradient([1;z2]);
    delta2 = delta2(2:end);
    Delta2 = Delta2 + delta3 * a2';
    Delta1 = Delta1 + delta2 * a1';
end

Theta1_grad(:,1) = Delta1(:,1)/m;
Theta1_grad(:,2:end) = Delta1(:,2:end)/m + (lambda/m) * Theta1(:,2:end);
Theta2_grad(:,1) = Delta2(:,1)/m;
Theta2_grad(:,2:end) = Delta2(:,2:end)/m + (lambda/m) * Theta2(:,2:end);

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
