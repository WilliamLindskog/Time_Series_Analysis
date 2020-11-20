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

%% Model with input u
%basicAnalysis(u, 100, 0.05);
%upw = randn(500,1);
%pqARMA(2,2,upw); % ARMA(1,2) gives reasonable coefficients 
data = iddata(u); 
p = 1; q = 2; 
arma_u = armax(data, [p q]);
rar = modFilt(arma_u.A, arma_u.C, data.y);

basicAnalysis(rar, 100, 0.05);
present(arma_u);

%ARMAupw = armax(upw, [1 2]);
%ypw = filter(ARMAupw.a, ARMAupw.c, y);

%% Pre-whitening
upw = filter(arma_u.A, arma_u.C, u);
ypw = filter(arma_u.A, arma_u.C, y);

%% CCF u and y
M = 40; stem(-M:M, crosscorr(upw, ypw, M)); 
title("Cross Correlation Function") 
xlabel("Lag")
hold on    
plot(-M:M, 2/sqrt(length(upw))*ones(1, 2*M+1), "--")
plot(-M:M, -2/sqrt(length(upw))*ones(1, 2*M+1), "--")
hold off

% Seems that appropriate model orders are d = 4, r = 2, s = 0

%% Model estimation for input (A2, A3, B, C3) 
d = 4; r = 2; s = 0;

A2 = [1 zeros(1,r)];
B = [zeros(1,d) zeros(1,0)];
Mi = idpoly([1], [B], [], [], [A2]);
Mi.Structure.B.Free = [zeros(1,d) ones(1,s)]; 
zpw = iddata(ypw, upw);
Mba2 = pem(zpw, Mi);
vhat = resid(Mba2, zpw); 

%plotCCF_Filip(upw, vhat.OutputData,40); 
%basicAnalysis(vhat.OutputData, 40, 0.05);

present(Mba2);

%% Evaluating the model type and order for the ARMA part (C1, A1)
x = y - filter(Mba2.F, Mba2.F, u);
plotCCF_Filip(x, y, 40);
basicAnalysis(x, 40, 0.05);
% based on analysis AR(1) seems reasonable to use 

%% Final model estimation 
A1 = [1 0]; A2 = [1 0 0]; B = [0 0 0 0 1]; C = [1];
Mi = idpoly(1, B, C, A1, A2);
z = iddata(y, u);
MboxJ = pem(z, Mi);
present(MboxJ)
ehat = resid(MboxJ, z);
plotCCF_Filip(u, ehat.OutputData, 40);
basicAnalysis(ehat.OutputData, 40, 0.05);

clear all
%% Hairdryer data
load('tork.dat')
tork = tork - repmat(mean(tork), length(tork), 1); 
y = tork(:,1); u = tork(:,2); z = iddata(y,u); 
plot(z(1:300))
% Seems like u is on-and-off input 

%% ARMA model generation

%% Prediction of ARMA-processes
load svedala
y = svedala; 
A = [1 -1.79 0.84]; C = [1 -0.18 -0.11];
[F3, G3] = polydiv(C,A,3); [F26, G26] = polydiv(C,A,26); 
yhat_3 = filter(G3, C, y); 
yhat_26 = filter(G26, C, y);

%% Prediction of ARMAX-processes
load sturup
A = [1 -1.49 0.57]; B = [0 0 0 0.28 -0.26]; C = [1}; 

%% Prediction of SARIMA-process
AS = [1 zeros(1, S-1) -1];


