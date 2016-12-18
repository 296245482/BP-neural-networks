function W = randInitializeWeights(L_in, L_out)
% Initialize W randomly so that we break the symmetry while
% training the neural network.


epsilon = 0.12;
W = rand(L_out, 1 + L_in) * 2 * epsilon - epsilon;


end
