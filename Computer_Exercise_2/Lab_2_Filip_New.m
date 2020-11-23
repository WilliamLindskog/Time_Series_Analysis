%% 2.1 ---------------------------------------
% 
% Generating Box-Jenkins data

rng(0)
n = 500;
A1 = [1 -0.65];
A2 = [1 .90 .78];
C = 1;
B = [0 0 0 0 .4];
e = sqrt(1.5) * randn(n+100, 1);
w = sqrt(2) * randn(n+100, 1);
A3 = [1 .5];
C3 = [1 -.3 .2];
u = filter(C3, A3, w);
y = filter(C, A1, e) + filter(B, A2, u);
u = u(101:end);
y = y(101:end);
clear A1 A2 C B e w A3 C3

%% 2.1 Building a model for the input u

%basicIdentification(u, 100, 0.05);
data = iddata(u);
p = 1;
q = 2;

arma_u = armax(data, [p q]);
rar = modFilter(arma_u.A, arma_u.C, data.y);

basicIdentification(rar, 100, 0.05);
present(arma_u);

clear p q rar 

%% 2.1 Pre-whitening
% modFilter?
upw = filter(arma_u.A, arma_u.C, u);
ypw = filter(arma_u.A, arma_u.C, y);

%% 2.1 CCF from upw to ypw

plotCCF(upw,ypw,40);

% It seems that appropriate model orders are d=4, r=2, s = 0

%% 2.1 Model estimation for the input part (A2, A3, B, C3)
d = 4;
r = 2;
s = 0;

A2 = [1 zeros(1,r)];
B = [zeros(1,d) 1 zeros(1,s)];
Mi = idpoly([1], B, [], [], A2);
Mi.Structure.B.Free = [zeros(1,d) ones(1,s+1)];
zpw = iddata(ypw, upw);
Mba2 = pem(zpw, Mi);
vhat = resid(Mba2, zpw);

plotCCF_Filip(upw, vhat.OutputData, 40);
basicIdentification(vhat.OutputData, 40, 0.05);

present(Mba2)


%% 2.1 Evaluatingthe model type and order for the ARMA part (C1, A1)

x = y - filter(Mba2.B, Mba2.F, u);

figure()
M = 40;
[ccf, lags, bounds] = crosscorr(x, u, M);
stem(lags, ccf);
title('Cross-correlation function'), xlabel('Lag');
hold on
yline(bounds(1), '--r');
yline(bounds(2), '--r');
xline(0, '--');
hold off

basicIdentification(x, 40, 0.05);
% Based on the ACF and PACF an AR(1) model seems approriate

%% 2.1 Final model estimation

A1 = [1 0];
A2 = [1 0 0];
B = [0 0 0 0 1];
C = [1];

Mi = idpoly(1, B, C, A1, A2);
z = iddata(y, u);
MboxJ = pem(z, Mi);

ehat = resid(MboxJ, z);

present(MboxJ);
plotCCF(u, ehat.OutputData, 40);
basicIdentification(ehat.OutputData, 40, 0.05);

%% 2.2 ---------------------------------
%
% 

load tork.dat

tork = tork - repmat(mean(tork), length(tork), 1);
y = tork(:,1);
u = tork(:,2);
z = iddata(y,u);


%% 2.2 Model for the input u

%basicIdentification(u, 100, 0.05); % The input seems to be roughly AR(1)

A3 = [1 0];
B = [];
C3 = [];

model_pw_init = idpoly(A3, B, C3);

model_pw = pem(u, model_pw_init);
rar = resid(z.InputData, model_pw);

basicIdentification(rar.OutputData, 100, 0.05);
present(model_pw);

clear A3 B C3 model_pw_init

%% 2.2 Pre-whitening 

upw = modFilter(model_pw.A, model_pw.C, u);
ypw = modFilter(model_pw.A, model_pw.C, y);

%% 2.2 CCF between upw and ypw
plotCCF(upw, ypw, 100);
% Appropriate model orders seems to be:
%       d = 3
%       r = 1
%       s = 2

%% 2.2 Model estimation for the input part
d = 2;
r = 2;
s = 2;

A2 = [1 zeros(1,r)];
B = [zeros(1,d) 1 zeros(1,s)];

model_inp_init = idpoly([1], B, [], [], A2);
model_inp_init.Structure.B.Free = [zeros(1,d) ones(1,s+1)];

zpw = iddata(ypw,upw);
model_inp = pem(zpw, model_inp_init);

vhat = resid(model_inp, zpw);

basicIdentification(vhat.OutputData, 40, 0.05);
plotCCF(upw, vhat.OutputData, 40);
present(model_inp)

%% 2.2 Modeling the ARMA part (A1, C1)

x = y-filter(model_inp.B, model_inp.F, u);
plotCCF(x, u, 40);
basicIdentification(x, 100, 0.05);

A1 = [1 0];
B = [];
C1 = [];

model_x_init = idpoly(A1, B, C1);

model_x = pem(x, model_x_init);
rar_x = resid(x, model_x);

basicIdentification(rar_x.OutputData, 100, 0.05);
present(model_x);
whitenessTest(rar_x.OutputData);

%% 2.2 Final model prediction

A1 = [1 0];
A2 = [1 0 0];
B = [0 0 1 0 0];
C = [];

model_final_init = idpoly(1, B, C, A1, A2);
model_final_init.Structure.B.Free = [0 0 1 1 1];

model_final = pem(z, model_final_init);
rar_final = resid(z, model_final);

present(model_final);
whitenessTest(rar_final.OutputData);
basicIdentification(rar_final.OutputData, 40, 0.05);
plotCCF(u, rar_final.OutputData, 40);


%% 2.3 --------------------------
%
%

load svedala

plot(svedala)

%% 2.3 Prediction with k=3 and k = 26

A = [1 -1.79 0.84];
C = [1 -0.18 -0.11];

k1 = 3;
k2 = 26;

[F1, G1] = polydiv(C, A, k1);
yhat_1 = modFilter(G1, C, svedala);
res_1 = svedala(3:end) - yhat_1;

[F2, G2] = polydiv(C, A, k2);
yhat_2 = modFilter(G2, C, svedala);
res_2 = svedala(3:end) - yhat_2;

mean(res_1)
mean(res_2)
%% Estimating noise variance using k=1

k = 1;

[F, G] = polydiv(C, A, k);

yhat = modFilter(G, C, svedala);

res = svedala(3:end) - yhat;

basicIdentification(res, 40, 0.05);
figure()
whitenessTest(res);

sigma2 = var(res)

%% 2.3 Confidence intervals for prediction errors

sign = 0.05;

conf_1 = norminv([sign/2 1-sign/2]).*sqrt(sigma2).*sqrt(sum(F1.^2));
conf_2 = norminv([sign/2 1-sign/2]).*sqrt(sigma2).*sqrt(sum(F2.^2));


%% Percentage of errors outside confidence interval

disp('Share of prediction errors outside confidence interval (k=3)')
disp([num2str((sum(abs(res_1) > conf_1(2))/length(res_1))*100), '%', newline])
disp('Share of prediction errors outside confidence interval (k=26):')
disp([num2str((sum(abs(res_2) > conf_2(2))/length(res_2))*100), '%'])


%% 2.3 Plots of processes, predictions and prediction errors

figure()
subplot(221)
hold on
plot(yhat_1, 'r')
plot(svedala, 'b')
legend('3 step estimation', 'True values')
subplot(222)
plot(res_1)
yline(conf_1(1), '--r');
yline(conf_1(2), '--r');
subplot(223)
hold on
plot(yhat_2, 'r')
plot(svedala, 'b')
legend('26 step estimation', 'True values')
subplot(224)
plot(res_2)
yline(conf_2(1), '--r');
yline(conf_2(2), '--r');

basicIdentification(res_1, 40, 0.05);
%% 2.4 ------------------------
%
%

