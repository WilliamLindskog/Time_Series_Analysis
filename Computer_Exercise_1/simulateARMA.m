function y = simulateARMA(C, A, sigma2, n, seed)
% Simulates n+100 samples and removes the initial 100
    if nargin == 5
        rng(seed)
    end
    
    e = sqrt(sigma2)*randn(n+100, 1);
    
    y = filter(C, A, e);
    y = y(101:end);
end

