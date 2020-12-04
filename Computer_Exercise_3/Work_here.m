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
   ls2(i) = sum((tar2 - yhat).^2); % Sum of squared error for specific lambda
end
plot(lambda_line,ls2) % Optimal lambda at 0.94

%% Estimate A-parameters recursively 

model = [2]; lambda = 0.9424;
[Aest, yhat, covAest, yprev] = rarx(tar2, model, 'ff', lambda); 
figure()
subplot(211)
plot(Aest)
subplot(212)
plot(thx)

%% 2.2 --------

y = tar2;

% Length data
N = length(y);

% Define the state space equations
A = [1 0; 0 1];
Re = 0.001*[1 0; 0 0];% Hidden state noise covariance 
Rw = 1.25; % Observation variance

% Set initial values 
Rxx_1 = 1*eye(2);
xtt_1 = [0 0];

% Vector to store values in
xsave = zeros(2,N); 
ysave = zeros(1,N);

% Kalman filter. Start from k = 3
% since we need old values of y
for k = 3:N
    % C is a function of time
    C = -[y(k-1) y(k-2)];
    
    % Update
    Ryy = C*Rxx_1*C.' + Rw;
    Kt = (Rxx_1*C.')/Ryy;
    xtt = xtt_1 + Kt*(y(k)-C.*xtt_1);
    Rxx = (eye(2) - Kt*C)*Rxx_1; 
    
    % Save 
    xsave(:,k) = xtt;
    ysave(k) = C*xtt_1;
    % Predict
    Rxx_1 = A*Rxx*A.' + Re;
    xtt_1 = A*xtt;
    
end

ls  = sum((ysave).^2); 

figure()
subplot(221)
hold on 
plot(xsave(1,:), 'b');
plot(thx(:,1), '--k');
subplot(222)
hold on
plot(xsave(2,:), 'r');
plot(thx(:,2), '--k');
subplot(2,2,[3 4])
hold on 
plot(ysave)
plot(y, '--k')

%% 2.3. -----------------------------

n = 1000; b = 20; var_e = 1; var_v = 4; 
P = [7/8 1/8; 1/8 7/8]; 
u = dtmc(P); 

for t = 2:n
   x(t) = x(t-1) + sqrt(var_e)*randn(1,1);
end

y = x + b.*u + v
plot(y); 

%% Kalman filter on y


%% recursive temperature modeling
load svedala94.mat
%plot(svedala94)

y = svedala94;


A6 = [1 zeros(1,5) -1];
ys = modFilt(A6, 1, y);
ys = ys(7:end); 
%plot(ys)
%whitenessTest(ys, 0.05, 100);
%basicAnalysis(ys, 100, 0.05);

T = linspace(datenum(1994, 1, 1), datenum(1994, 12,31), length(y));
plot(T(7:end), ys);
datetick('x');


th = armax(ys, [2 2]);
th_winter = armax(ys(1:540), [2 2]); % A = [1,-1.666,0.706];  C = [1,-0.833,-0.107]; 
th_summer = armax(ys(907:1458), [2 2]); % A = [1,0.254,-0.459]; C = [1,1.032,0.0427]; 

% Summer temperature varies much more, day and night. Parameters are
% different from each other

lambda = 0.999;

th0 = [th_winter.A(2:end) th_winter.C(2:end)];
[thr, yhat] = rarmax(ys, [2 2], 'ff', lambda, th0);

subplot(311)
plot(T, y);
datetick('x')

subplot(312)
plot(thr(:,1:2))
hold on
plot(repmat(th_winter.A(2:end),[length(thr) 1]), 'b:');
plot(repmat(th_summer.A(2:end),[length(thr) 1]), 'r:');
axis tight
hold off

subplot(313)
plot(thr(:,3:end))
hold on
plot(repmat(th_winter.C(2:end),[length(thr) 1]), 'b:');
plot(repmat(th_summer.C(2:end),[length(thr) 1]), 'r:');
axis tight
hold off

% Since the recursive paramters do not change, it seems that th_winter and th_summer are similar processes
% recursive process seems to coincide with
% th_summer. 

%% Recursive temperature modeling again

load svedala94
y = svedala94(850:1100);
y = y - mean(y); 

%% Form the model

t = (1:length(y))';
U = [sin(2*pi*t/6) cos(2*pi*t/6)]; % change in season results in higher or lower amplitude (smaller coefficients) 
Z = iddata(y, U);
model = [3 [1 1] 4 [0 0]]; %[na [nb_1 nb_2] nc [nk_1 nk_2]]

thx = armax(Z, model);

%% Plot model and seasonal function

figure()
subplot(211)
plot(y)
subplot(212)
plot(U.*cell2mat(thx.b))

rar_thx = resid(Z, thx); 

basicIdentification(rar_thx, 100, 0.05);

%%  Constant external signal

U = [sin(2*pi*t/6) cos(2*pi*t/6) ones(size(t))];
Z = iddata(y, U); 
m0 = [thx.A(2:end) cell2mat(thx.B) 0 thx.C(2:end)];
Re = diag([0 0 0 0 0 1 0 0 0 0]);
model = [3 [1 1 1] 4 0 [0 0 0] [1 1 1]]; 
[thr, yhat] = rpem(Z, model, 'kf', Re, m0); 

%% Reconstruct

m = thr(:, 6);
a = thr(end, 4); 
b = thr(end, 5);
y_mean = m + a*U(:,1) + b*U(:, 2);
y_mean = [0; y_mean(1:end-1)];

%% Plot

figure()
hold on
plot(y)
plot(y_mean)
hold off

%% Study the entire year

y = svedala94; 
y = y - y(1); 
y = y - mean(y);

t = (1:length(y))';
U = [sin(2*pi*t/6) cos(2*pi*t/6)]; % change in season results in higher or lower amplitude (smaller coefficients) 
Z = iddata(y, U);
model = [3 [1 1] 4 [0 0]]; %[na [nb_1 nb_2] nc [nk_1 nk_2]]

thx = armax(Z, model);

U = [sin(2*pi*t/6) cos(2*pi*t/6) ones(size(t))];
Z = iddata(y, U); 
m0 = [thx.A(2:end) cell2mat(thx.B) 0 thx.C(2:end)];
Re = 1*diag([0 0 0 0 0 1 0 0 0 0]);
model = [3 [1 1 1] 4 0 [0 0 0] [1 1 1]]; 
[thr, yhat] = rpem(Z, model, 'kf', Re, m0); 

m = thr(:, 6);
a = thr(end, 4); 
b = thr(end, 5);
y_mean = m + a*U(:,1) + b*U(:, 2);
y_mean = [0; y_mean(1:end-1)];

figure()
hold on
plot(y)
plot(y_mean)
hold off

mse = sum((y - y_mean).^2)/length(y); % average 5 degrees difference

% Both function follow each other similiarly, seems to be a good estimate
% of the mean. 

