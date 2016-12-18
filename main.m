clear;
close all;
clc;

input_layer_size  = 256;   % 1x35 Input Images of Digits
hidden_layer_size = 200;   % 25 hidden units
num_labels = 10;          % 10 labels, from 1 to 10 (note that we have mapped "0" to label 10)
                          
% Load Training Data
fprintf('Loading Data ...\n');
load('semeion.data');
[sample_nums, sample_dims] = size(semeion);

train_nums = round(sample_nums * 0.75);

%random the dataset
[m,n] = sort(rand(1,sample_nums));
semeion = semeion(n(1:sample_nums),:);

X = semeion(1:train_nums,1:sample_dims - num_labels);
temp_y = zeros(sample_nums,1);
for i = 1:sample_nums
    for j = sample_dims - num_labels + 1:sample_dims
        if semeion(i,j) == 1
            temp_y(i,1) = j - input_layer_size;
        end
    end
end
y = temp_y(1:train_nums,:);
data_test = semeion(train_nums + 1:sample_nums,1:input_layer_size);
label_test = temp_y(train_nums + 1:sample_nums,:);

%Initializing Neural Network Parameters
fprintf('\nInitializing Neural Network Parameters ...\n');
initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

%Training Neural Network
fprintf('\nTraining Neural Network... \n')
options = optimset('MaxIter', 50);
lambda = 1;

costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, lambda);

[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

%get Theta1 and Theta2
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

%test accuracy             
pred = predict(Theta1, Theta2, data_test);
fprintf('\nAccuracy For Testing Dataset: %f\n', mean(double(pred == label_test)) * 100);
pred = predict(Theta1, Theta2, X);
fprintf('\nAccuracy For Training Dataset: %f\n', mean(double(pred == y)) * 100);

%GUI for test
save myparams Theta1 Theta2;
fprintf('press any key to GUI\n\n');
pause;

%call GUI function to visualized the result
GUI;