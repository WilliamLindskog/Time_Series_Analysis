%% Generating Box-Jenkins data
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

%% Building a model for the input u

%basicIdentification(u, 100, 0.05);
data = iddata(u);
p = 1;
q = 2;

arma_u = armax(data, [p q]);
rar = modFilter(arma_u.A, arma_u.C, data.y);

basicIdentification(rar, 100, 0.05);
present(arma_u);

clear p q rar data

%% Pre-whitening
% modFilter?
upw = filter(arma_u.A, arma_u.C, u);
ypw = filter(arma_u.A, arma_u.C, y);

%% CCF from upw to ypw

plotCCF(upw,ypw,40);

% It seems that appropriate model orders are d=4, r=2, s = 0

%% Model estimation for the input part (A2, A3, B, C3)
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

plotCCF(upw, vhat.OutputData, 40);
basicIdentification(vhat.OutputData, 40, 0.05);

present(Mba2)


%% Evaluatingthe model type and order for the ARMA part (C1, A1)

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

%% Final model estimation

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
