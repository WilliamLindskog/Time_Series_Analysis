%% Preparation for lab 3 (a)
prob1 = 7/8; prob2 = 1/8;
P = [prob1 prob2; prob2 prob1];
u = dtmc(P);
figure()
graphplot(u)

%% AR(2) state space form + Kalman filter

A = [-a1 1; -a2 0]; B = [1; 0]; C = [1 0];
%model = ssm(A, B, C); 


%% Estimate parameters using Kalman filter

y(t) = 