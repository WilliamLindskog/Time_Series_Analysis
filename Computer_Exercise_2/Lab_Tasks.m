%% Generating data, clearing unnecessary variables
n = 500; A1 = [1 -.65]; A2 = [1 .90 .78]; C = 1; B = [0 0 0 0 .4]; A3 = [1 .5]; C3 = [1 -.3 .2]; 
[u, y] = BJmodel(0, n, A1, A2, C, B, A3, C3);
clearvars A1 A2 C B e w A3 C3
subplot(211)
plot(u)
title("Input")
subplot(212)
plot(y)
title("Output")
arma_model_u = armax(u, [1 2]);

%% Determine orders of B and A polynomials
upw = randn(500,1);
%pqARMA(2,2,upw); % ARMA(1,2) gives reasonable coefficients 
ARMAupw = armax(upw, [1 2]);
ypw = filter(ARMAupw.a, ARMAupw.c, y);

%% CCF u and y
M = 40; stem(-M:M, xcorr(upw, ypw, M)); 
title("Cross Correlation Function") 
xlabel("Lag")
hold on    
plot(-M:M, 2/sqrt(length(upw))*ones(1, 2*M+1), "--")
plot(-M:M, -2/sqrt(length(upw))*ones(1, 2*M+1), "--")
hold off
H = xcorr(upw, ypw, M);
%% Pre-whiten y(t)