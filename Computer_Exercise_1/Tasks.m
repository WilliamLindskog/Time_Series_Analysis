%% Creating 2 ARMA polynomials
% Customized functions for this script can be found in the bottom. I have
% been using them regularly througout the script. 

[A1, C1] = polynomial(1);
[A2, C2] = polynomial(2); 
ARMA_poly1 = idpoly(A1, [], C1);
ARMA_poly2 = idpoly(A2, [], C2);

%% Simulate y1 and y2 processes
N = 300; var = 1.5;
y1 = simulate(ARMA_poly1, N, var, 0); 
y2 = simulate(ARMA_poly2, N, var, 0);

% PZ-plot and realisations
figure(1)
sgtitle('Pole-zero plots and realizations')
subplot(221)
pzmap(ARMA_1)
title('ARMA 1')
subplot(222)
pzmap(ARMA_2)
title('ARMA 2')
subplot(223)
plot(y1)
subplot(224)
plot(y2)
% y2 diverges as one pole lies outside of unit circle

%% Theoretical vs estimated covariance 
sigma2 = 1; m = 20;
theoEstDiff(ARMA_poly1, y1, sigma2, m); % Not identical as we do not have infinite data points for simulating the covariance function

%% Re-estimation y1
m = 20; identify(y1, m, 0.05); 
data1 = iddata(y1); 
ARpModel(5, data1); % 

%% Create ARMA Model
p = 1; q = 2; am11_model = armax(y1, [p q]);
e_hat = modFilt(am11_model, y1);
identify(e_hat, 20, 0.05);
present(am11_model);

%% Task 2.2. Model order estimation of an ARMA-process
%
clear

load data.dat
load noise.dat
data = iddata(data); % 
ARpModel(5, data); % Seems like p = 4 or 5 gives best result, considering FPE, MSE and Fit to estimation data

%% ARMA(p,q) model
% ARMA(2,1) has the lowest FPE
pMax = 2; qMax = 2; pqARMA(pMax, qMax, data, noise); 

%% Comparing AR(4) and ARMA(2,1)
ar_model = arx(data, [4]);
arma_model = armax(data, [2 1]);

rar_ar = resid(ar_model, data);
rar_arma = resid(arma_model, data); % ARMA(2,1) has lower FPE

identify(rar_ar.y, 40, 0.05);
identify(rar_arma.y, 40, 0.05);

%% Estimation of a SARIMA-process
clear all 

s = 12; A = [1 -1.5 0.7]; C = [1 zeros(1,s-1) -0.5]; e = randn(10000,1);
y = estSARIMA(s, A, C, e); 
%plot(y);
identify(y, 30, 0.05);

%% Removing the season 
A_season = [1 zeros(1, s - 1),-1];
y_s = filter(A_season, 1, y);
y_s = y_s(length(A_season):end);
data = iddata(y_s);

%% Initial Modelling 
A = [1 0 0]; B = []; C = [1 zeros(1, s)]; 
model_init = idpoly(A, B, C); 
model_init.Structure.c.Free = [zeros(1, s), 1];
model_init.Structure.A.Free = [0 1 1];

model_armax = pem(data, model_init); 
rar_armax = resid(model_armax, data);

plot(rar_armax.OutputData)
basicIdentification(rar_armax.OutputData, 30, 0.05)
present(model_armax)
res_variance = var(rar_armax.OutputData)

%% Not removing trend
clear all 

s = 12; A = [1 -1.5 0.7]; C = [1 zeros(1,s-1) -0.5]; e = randn(10000,1);
y = estSARIMA(s, A, C, e); 
%plot(y);
identify(y, 30, 0.05); % Higher FPE and MSE, using more degrees of freedom 
%% Estimation on Real Data
clear all

load svedala

data = iddata(svedala);

A = [1 zeros(1,25)]; B = []; C = [1 zeros(1,24)];

model_init = idpoly(A,B,C);
model_init.Structure.A.Free = [0 1 1 zeros(1,20) 1 1 1];
model_init.Structure.C.Free = [zeros(1,24) 1];

model_armax = pem(data, model_init);
rar_armax = resid(model_armax, data);

plot(rar_armax.OutputData)
basicIdentification(rar_armax.OutputData, 60, 0.05)
present(model_armax)
res_variance = var(rar_armax.OutputData)

%% Functions 
% Function handling what set to use (A,C)- polynomials
function [A,C] = polynomial(n)
    if n == 1
        A = [1 -1.79 0.84];
        C = [1 -0.18 -0.11];
    elseif n == 2
        A = [1 -1.79];
        C = [1 -0.18 -0.11];
    else
        error('Insert value 1 or 2 in Polynomial function');
    end
end

% Simulate function
function s = simulate(poly, N, sigma2, seed)
if nargin == 4
    rng(seed)
end
s = filter(poly.c, poly.a, sqrt(sigma2)*randn(N,1));
s = s(101:end); 
end

function theoEstDiff(poly, y, sigma2, m)
r_theo = kovarians(poly.c, poly.a, m);
stem(0:m, r_theo*sigma2)
hold on
r_est = covf(y, m + 1);
stem(0:m, r_est, 'r')
end

% Function plotting AR processes of different orders
function ARpModel(p, Data)
rng(0)
figure()
for p = 1:5
    ar_model = arx(Data, p);
    rar = resid(ar_model, Data);
    
    present(ar_model);
    
    subplot(1,5,p)
    hold on
    plot(rar)
    %plot(Noise, 'r')
end
end

% Function showing acf and pacf in same figure
function [ACF, PACF] = identify(est,maxOrder,signf)
figure()
subplot(131)
acf(est, maxOrder, signf, 1);
title('ACF')
subplot(132)
pacf(est, maxOrder, signf, 1);
title('PACF')
subplot(133)
normplot(est)
end

% Modified filter function 
function e_hat = modFilt(poly, y)
e_hat = filter(poly.a, poly.c, y);
e_hat = e_hat(length(poly.a:end));
end

function pqARMA(pMax, qMax, data, noise)
for p = 1:pMax
    for q = 1:qMax
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
end

% Estimation and plot of SARIMA process
function y = estSARIMA(season, A, C, e)
A_season = [1 zeros(1,season - 1) -1];
A_star = conv(A, A_season);
y = filter(C, A_star, e);
y = y(101:end);
end