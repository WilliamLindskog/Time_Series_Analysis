function pqARMA(pMax, qMax, data)
for p = 1:pMax
    for q = 1:qMax
        arma_model = armax(data, [p q]);
        %rar = resid(arma_model, data);
        present(arma_model);
        basicAnalysis(data, 100, 0.05);
        %subplot(2,2,2*p+q-2)
        %hold on
        %plot(rar)
        %title(['(',num2str(p),',',num2str(q),')']);
    end
end
end