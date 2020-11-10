%% 2.1
% When simulating the processes, the second ARMA process diverges. This is
% because it has a pole outside the unit circle

% ARMA processes
A1 = [1 -1.79 0.84];
C1 = [1 -0.18 -0.11];

A2 = [1 -1.79];
C2 = [1 -0.18 -0.11];

ARMA_1 = idpoly(A1,[],C1);
ARMA_2 = idpoly(A2,[],C2);

%% Output simulation
n = 200;
sigma2 = 1;

y_1 = simulateARMA(ARMA_1.C, ARMA_1.A, sigma2, n, 0);
y_2 = simulateARMA(ARMA_2.C, ARMA_2.A, sigma2, n, 0);

% Plots: pole-zero and realizations
figure(1)
sgtitle('Pole-zero plots and realizations')
subplot(221)
pzmap(ARMA_1)
title('ARMA 1')
subplot(222)
pzmap(ARMA_2)
title('ARMA 2')
subplot(223)
plot(y_1)
subplot(224)
plot(y_2)

%% Estimated vs theoretical covariance functions for y_1
% They differ since we do not have infinite data points for simulating the
% covariance functions
m = 20;
r_theo_1 = kovarians(ARMA_1.C, ARMA_1.A, m);
r_est_1 = covf(y_1, m+1);

% Plots: 
figure()
hold on
stem(0:m, r_theo_1*sigma2, 'black')
stem(0:m, r_est_1, 'r')
legend('Theoretical covariance function', 'Estimated covariance function')

%% Re-estimating y_1

m = 20;

basicIdentification(y_1, m, 0.05);

data = iddata(y_1);

% Building an ARMA(p,q) model
p = 2;
q = 2;

arma_model = armax(y_1, [p q]);
e_hat = modFilter(arma_model.A, arma_model.C, y_1);

basicIdentification(e_hat, 20, 0.05);

present(arma_model)

%% 2.2 Model order estimation of an ARMA-process
%
%
clear

load data.dat
load noise.dat
data = iddata(data);

%% AR(p) model
% AR(4) has the lowest FPE
figure()
for p = 1:5
    ar_model = arx(data, [p]);
    rar = resid(ar_model, data);
    
    present(ar_model);
    
    subplot(1,5,p)
    hold on
    plot(rar)
    plot(noise, 'r')
end

%% ARMA(p,q) model
% ARMA(2,1) has the lowest FPE
figure()
for p = 1:2
    for q = 1:2
        
        arma_model = armax(data, [p q]);
        rar = resid(arma_model, data);

        present(arma_model);

        subplot(2,2,2*p+q-2)
        hold on
        plot(rar)
        plot(noise, 'r')
        title(['(',num2str(p),',',num2str(q),')']);
    end
end

%% Comparing AR(4) and ARMA(2,1)
ar_model = arx(data, [4]);
arma_model = armax(data, [2 1]);

rar_ar = resid(ar_model, data);
rar_arma = resid(arma_model, data);

basicIdentification(rar_ar.y, 40, 0.05);
basicIdentification(rar_arma.y, 40, 0.05);

%% 2.3  Estimation of a SARIMA
% Simulation and basic analysis of data
rng(0)
A = [1 -1.5 0.7];
C = [1 zeros(1,11) -0.5];
A12 = [1 zeros(1,11) -1];
A_star = conv(A, A12);
e = randn(600, 1);
y = modFilter(C, A_star, e);
y = y(101:end);

plot(y)
basicIdentification(y, 30, 0.05);
%% Building SARIMA model
y_s = modFilter(A12, 1, y);
y_s = y_s(length(A12):end);

data = iddata(y_s);

A = [1 0 0];
B = [];
C = [1 zeros(1,12)];

model_init = idpoly(A, B, C);
model_init.Structure.C.Free = [zeros(1,12) 1];
model_init.Structure.A.Free = [0 1 1];

model_armax = pem(data, model_init);
rar_armax = resid(model_armax, data);

plot(rar_armax.OutputData)
basicIdentification(rar_armax.OutputData, 30, 0.05)
present(model_armax)
res_variance = var(rar_armax.OutputData)

%% 2.4 Estimation on real data

load svedala

data = iddata(svedala);

A = [1 zeros(1,25)];
B = [];
C = [1 zeros(1,24)];

model_init = idpoly(A,B,C);
model_init.Structure.A.Free = [0 1 1 zeros(1,20) 1 1 1];
model_init.Structure.C.Free = [zeros(1,24) 1];

model_armax = pem(data, model_init);
rar_armax = resid(model_armax, data);

plot(rar_armax.OutputData)
basicIdentification(rar_armax.OutputData, 60, 0.05)
present(model_armax)
res_variance = var(rar_armax.OutputData)