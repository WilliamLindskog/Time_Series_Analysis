function [outputArg1,outputArg2] = plotCCF_Filip(x, y, M)
figure()
    [ccf, lags, bounds] = crosscorr(x, y, M);
    stem(lags, ccf);
    title('Cross-correlation function'), xlabel('Lag');
    hold on
    yline(bounds(1), '--r');
    yline(bounds(2), '--r');
    xline(0, '--');
    hold off
end

