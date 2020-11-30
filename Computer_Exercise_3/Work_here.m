%% Load and plot tar2 and thx data
load tar2.dat
load thx.dat
figure()
subplot(211)
plot(tar2)
subplot(212)
plot(thx)

%% Choose lambda

n = 100; 
lambda_line = linspace(0.85, 1, n);
ls2 = zeros(n,1); 
for i = 1:length(lambda_line)
   [Aest, yhat, covAest, trash] = rarx(tar2, [2], 'ff', lambda_line(i));
   ls2(i) = sum((tar2 - yhat).^2); 
end
plot(lambda_line,ls2) % Optimal lambda at 0.94

%% Estimate A-parameters recursively 

model = [2]; lambda = 0.94;
[Aest, yhat, covAest, yprev] = rarx(tar2, model, 'ff', lambda); 
figure()
subplot(211)
plot(Aest)
subplot(212)
plot(thx)

