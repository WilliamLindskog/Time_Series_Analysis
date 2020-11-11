%% Generating data
n = 500; A1 = [1 -.65]; A2 = [1 .90 .78]; C = 1; B = [0 0 0 0 .4]; A3 = [1 .5]; C3 = [1 -.3 .2]; 
[u, y] = BJmodel(0, n, A1, A2, C, B, A3, C3);

%% Determine orders
