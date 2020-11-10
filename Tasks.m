%% Creating 2 ARMA polynomials
[A1 C1] = polynomial(1);
ARMA_poly1 = idpoly(A1, [], C1);

[A2 C2] = polynomial(2); 
ARMA_poly2 = idpoly(A2, [], C2);

%% View poles and zeros for ARMA objects
pzmap(ARMA_poly1);
%pzmap(ARMA_poly2); %Pole outside of unit circle

%% Simulate ARMA process
simData = simulate(ARMA_poly1, 300, 1.5);

%% Simulate y1 and y2 processes
y1 = simulate(ARMA_poly1, 300, 1.5); 
y2 = simulate(ARMA_poly2, 300, 1.5);
subplot(211)
plot(y1)
subplot(212)
plot(y2) % y2 diverges as one pole lies outside of unit circle

%% Theoretical vs estimated covariance 
m = 20;
sigma2 = 1;
r_theo = kovarians(ARMA_poly1.c, ARMA_poly1.a, m);
stem(0:m, r_theo*sigma2)
hold on
r_est = covf(y1, m + 1)
stem(0:m, r_est,'r') % Why aren't they identical????

%% Re-estimate model using data vector y1 
plotACFnPACFnNorm(y1, 100, 0.05, true);

%% Estimation 
data1 = iddata(y1); 
e1 = modFilt(ARMA_poly1.a, ARMA_poly1.c, y1)
e2 = modFilt(ARMA_poly1.a, ARMA_poly1.c, y2)

%% Task 2.2. Model order estimation of an ARMA-process
load data.dat
data = iddata(data);
plotAR(5, data)

%% Model data using ARMA(p,q)- models
p = 1;
q = 2;
am11_model = armax(data,[p q])

%% Estimation of a SARIMA-process
s = 12;
A = [1 -1.5 0.7];
C = [1 zeros(1,s-1) -0.5];
e = randn(600,1);
y = estSARIMA(s, A, C, e); 
plotACFnPACFnNorm(y,100, 0.05,true);

%% Removing the season 
A_season = [1 zeros(1, s - 1),-1];
y_s = filter(A_season, 1, y);
y_s = y_s(length(A_season):end);
data = iddata(y_s);

%% Initial Modelling 
A = [1 0 0]; B = []; C = []; 
model_init = idpoly(A, B, C); 
model_armax = pem(data, model_init); 

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
function s = simulate(p, N, sigma2)
rng(0)
s = filter(p.c, p.a, sqrt(sigma2)*randn(N,1));
s = s(101:end); 
end

% Function plotting AR processes of different orders
function plotAR(p, Data)
if rem(p, 2) == 0
    for i = 1:p
        subplot(p/2,p/2,i);
        rar = resid(arx(Data,i),Data);
        present(arx(Data,i));
        plot(rar);
    end
else
    for i = 1:p
        subplot(p/2 - 1/2,p/2 + 1/2,i);
        rar = resid(arx(Data,i),Data);
        present(arx(Data,i));
        plot(rar);
    end
end
end

% Function showing acf and pacf in same figure
function plotACFnPACFnNorm(y,maxOrd,signf,show)
subplot(2,2,1)
acf(y, maxOrd, signf, show);
subplot(2,2,2)
pacf(y, maxOrd, signf, show);
subplot(2,2,3)
normplot(y);
end

% Modified filter function 
function e_hat = modFilt(a, c, y)
e_hat = filter(a, c, y);
e_hat = e_hat(length(a:end));
end

% Estimation and plot of SARIMA process
function y = estSARIMA(season, A, C, e)
A_season = [1 zeros(1,season - 1) -1];
A_star = conv(A, A_season);
y = filter(C, A_star, e);
y = y(101:end);

end