data = load('winequality.txt');% regression dataset % https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/
data = horzcat(data(:,12),data(:,1:11));
train_data = data(1:3000,:); % 3000 samples as training
test_data = data(3001:4898,:); % rest of training 
[TrainingTime,TestingTime,TrainingRMSE, TestingRMSE]=IPSO_ELM(train_data, test_data,0 , 10, 'sigmoid'); % Call Incremental Elm
