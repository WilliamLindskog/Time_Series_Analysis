%% Preparation for lab 3 (a)
prob1 = 7/8; prob2 = 1/8;
P = [prob1 prob2; prob2 prob1];
u = dtmc(P);
figure()
graphplot(u)

%% 