%% Generating data, clearing unnecessary variables
n = 500; A1 = [1 -.65]; A2 = [1 .90 .78]; C = 1; B = [0 0 0 0 .4]; A3 = [1 .5]; C3 = [1 -.3 .2]; 
[u, y] = BJmodel(0, n, A1, A2, C, B, A3, C3);
clearvars A1 A2 C B e w A3 C3 n
doublePlot(u,y)
arma_model_u = armax(u, [1 2]);
    
%% Model with input u (analysis)
basicAnalysis(u, 100, 0.05);
pqARMA(2,2,u); % ARMA(1,2) gives reasonable coefficients (WHY?)

%% Model with input u
data = iddata(u); 
p = 1; q = 2; 
arma_u = armax(data, [p q]);
rar = modFilt(arma_u.A, arma_u.C, data.y);

basicAnalysis(rar, 100, 0.05);
present(arma_u);

%ARMAupw = armax(upw, [1 2]);
%ypw = filter(ARMAupw.a, ARMAupw.c, y);
clear p q rar
%% Pre-whitening
upw = filter(arma_u.A, arma_u.C, u);
ypw = filter(arma_u.A, arma_u.C, y);
plotCCF(upw, ypw, 40);

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

plotCCF(upw, vhat.OutputData,40); 
%basicAnalysis(vhat.OutputData, 40, 0.05);

present(Mba2); % Is this really uncorrelated?

%% Evaluating the model type and order for the ARMA part (C1, A1)
x = y - filter(Mba2.F, Mba2.F, u);
plotCCF(x, y, 40);
basicAnalysis(x, 40, 0.05);
% based on analysis AR(1) seems reasonable to use 

%% Final model estimation 
A1 = [1 0]; A2 = [1 0 0]; B = [0 0 0 0 1]; C = [1];
Mi = idpoly(1, B, C, A1, A2);
z = iddata(y, u);
MboxJ = pem(z, Mi);
present(MboxJ)
ehat = resid(MboxJ, z);
plotCCF(u, ehat.OutputData, 40);
basicAnalysis(ehat.OutputData, 40, 0.05);

clear all

%% Hairdryer data (2.2) -------------------

load('tork.dat')

tork = tork - repmat(mean(tork), length(tork), 1); 
y = tork(:,1); u = tork(:,2); z = iddata(y,u); 
%plot(z(1:300))
% Seems like u is on-and-off input 

%basicAnalysis(u, 100, 0.05);

% seems to be roughly an AR(1)-process

A3 = [1 0]; B =[]; C3 = []; 
model_pw_init = idpoly(A3, B, C3);
model_pw = pem(u, model_pw_init); 
rar = resid(z.InputData, model_pw);

basicAnalysis(rar.OutputData, 100, 0.05);
present(model_pw);

clear A3 B C3 model_pw_init


%% Prediction of ARMA-processes (2.3) ----------------

load svedala
plot(svedala)

%% Prediction k = 3 & k = 26

A = [1 -1.79 0.84]; C = [1 -0.18 -0.11];
[F3, G3] = polydiv(C,A,3); [F26, G26] = polydiv(C,A,26); 

yhat_3 = modFilt(G3, C, svedala);
res_3 = svedala(3:end) - yhat_3; 

yhat_26 = modFilt(G26, C, svedala);
res_26 = svedala(3:end) - yhat_26;

%% Necessary variables for question
m3 = mean(res_3);
m26 = mean(res_26);
var_res_3 = var(res_3);
var_res_26 = var(res_26);

%% (2.3) Estimating noise variance using k = 1

k = 1; 

[F, G] = polydiv(C, A, k);

yhat = modFilt(G, C, svedala); 
res = svedala(3:end) - yhat; 

basicAnalysis(res, 100, 0.05);
figure()
whitenessTest(res);
sigma2 = var(res);

%% 2.3 Confidence intervals for prediction error

sign = 0.05; 

conf_1 = norminv([sign/2 1-sign/2]).*sqrt(sigma2).*sqrt(sum(F3.^2));
conf_2 = norminv([sign/2 1-sign/2]).*sqrt(sigma2).*sqrt(sum(F26.^2));

%% Percentage of errors outside confidence interval

disp('Share of prediction errors outside confidence interval (k=3)')
disp([num2str((sum(abs(res_3) > conf_1(2))/length(res_3))*100), '%', newline])
disp('Share of prediction errors outside confidence interval (k=26):')
disp([num2str((sum(abs(res_26) > conf_2(2))/length(res_26))*100), '%'])

%% 2.3 Plots of processes, predictions and prediction errors

figure()
subplot(221)
hold on
plot(yhat_3, 'r')
plot(svedala, 'b')
legend('3 step estimation', 'True values')
subplot(222)
plot(res_3)
yline(conf_1(1), '--r');
yline(conf_1(2), '--r');
subplot(223)
hold on
plot(yhat_26, 'r')
plot(svedala, 'b')
legend('26 step estimation', 'True values')
subplot(224)
plot(res_26)
yline(conf_2(1), '--r');
yline(conf_2(2), '--r');

basicIdentification(res_3, 40, 0.05);
clear all 
%% (2.4) Prediction of ARMAX-processes

load sturup
A = [1 -1.49 0.57]; B = [0 0 0 0.28 -0.26]; C = [1]; % delay = 3

%% Prediction
d = 3; k =1; 
[F, G] = polydiv(C,A,k); 

BF = conv(B,F); 
[Fhat, Ghat] = polydiv(BF,C,k); 

%[F3, G3] = polydiv(C, A, 3);
%BF = conv(B,F);
%[C, BF] = equalLength(C,BF);
%[Fhat, Ghat] = deconv(conv([1 zeros(1, d-1)], BF), C);

%yhat_3 = modFilt(G3, C, sturup);
%res_3 = sturup(3:end) - yhat_3; 

%[F26, G26] = polydiv(C, B, 26); 
%yhat_26 = modFilt(G26, C, sturup);
%res_26 = sturup(3:end) - yhat_26;


%%
clear all

%% (2.5) Prediction of SARIMA-process

load svedala
plot(svedala(1:100))
S = 24;

AS = [1 zeros(1, S-1) -1];
A = [1 -1.49 0.57]; C = [1]; 

A_star = conv(A, AS);
e = randn(1361,1);
y = modFilt(C, A_star, e);
%plot(y)
%basicAnalysis(y, 100, 0.05);

y_s = modFilt(AS, 1, y);
data = iddata(y_s);

A = [1 0 0]; B = []; C = [1 zeros(1, S)]; 
model_init = idpoly(A, B, C); 
model_init.Structure.C.Free = [zeros(1, S), 1];
model_init.Structure.A.Free = [0 1 1];

model_armax = pem(data, model_init); 
rar_armax = resid(model_armax, data);

plot(rar_armax.OutputData)
basicAnalysis(rar_armax.OutputData, 30, 0.05)
present(model_armax)
res_variance = var(rar_armax.OutputData)