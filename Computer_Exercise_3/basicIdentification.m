function [ACF, PACF] = basicIdentification(y, m, sign)
    figure()
    subplot(131)
    acf(y, m, sign, 1);
    title('ACF')
    subplot(132)
    pacf(y, m, sign, 1);
    title('PACF')
    subplot(133)
    normplot(y)
end

