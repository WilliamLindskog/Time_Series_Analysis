function x = kalmanFilter(y, A, C, Re, Rw, init_Rxx, init_xtt)
    N = length(y);
    
    sz = size(y); 
    xsave = zeros(sz(1),N);
    
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