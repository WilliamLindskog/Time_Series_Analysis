function [u,y] = BJmodel(seed, N, A1, A2, C, B, A3, C3)
if nargin == 8
    rng(seed)
end

e = sqrt(1.5)*randn(N + 100, 1);
w = sqrt(2) * randn(N + 100, 1);

u = filter(C3, A3, w);
y = filter(C, A1, e) + filter(B, A2, u);
u = u(101:end); y = y(101:end);
end