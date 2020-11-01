
clear all;clc; % clear output
%seed=2e5;
%rand('seed',seed);
%============================== Data Preprocessing============================================

Data = readtable('sonar.txt');% https://archive.ics.uci.edu/ml/datasets/Connectionist+Bench+(Sonar,+Mines+vs.+Rocks)
Data = Data(randperm(size(Data,1)),:); % Shuffle dataset
Data = table2cell(Data);
for i=1:208, % Give ALphabate value to integer
    Data(i,61) = cellfun(@double,Data(i,61),'uni',0);
end
Data = cell2table(Data);
for i=1:208, % Give Replace Label R with 1 and M with -1
    if table2array(Data(i, 61)) == 82,
        Data{i,61} = -1;
    else
        Data{i,61} = 1;
    end
end
data = Data{:,:};
data = horzcat(data(:,61),data(:,1:60));
train_data = data(1:125,:);
test_data = data(125:208,:);

[TrainingTime,TestingTime,TrainingAccuracy, TestingAccuracy]=IPSO_ELM(train_data, test_data, 1, 10, 'sigmoid');
