lfunction fit  = fitness(x,P,NumberofInputNeurons,NumberofTrainingData,E) % FITNESS FUNCTION FOR pso
    InputWeight=x(1,NumberofInputNeurons); %  w = rand(1,noofin)
    BiasofHiddenNeurons=x(NumberofInputNeurons+1);
    ind=ones(1,NumberofTrainingData);
    BiasMatrix=BiasofHiddenNeurons(:,ind);% bias
    H=(1 ./ (1 + exp(-(InputWeight*P+BiasMatrix)))); % H =  (1 / 1 + exp  (w *P + bias)
    %clear p;
    tempH = pinv(H'); % Temph =  H^-1
    B= tempH * E'; % Beta = H^-1 * E
    E = E - (H'*B)';% E = E - H*beta
    fit=sqrt(mse(E));
end