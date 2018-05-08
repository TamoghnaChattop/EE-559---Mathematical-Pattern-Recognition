% basic sequential GD

load('synthetic1.mat');
% use augmented space and reflect training data points
N = size(feature_train,1);
D = size(feature_train,2);
traindata = ones(N,D+1);
traindata(:,2)=feature_train(:,1);
traindata(:,3)=feature_train(:,2);
for i=1:N
    if (label_train(i)==2)
        traindata(i,:)= -traindata(i,:);
    end
end

% initialize w(0)
w=0.1*ones(N,D+1);

% randomly suffle the order of trainning data points
shufdata = traindata(randperm(N),:);

% do iterations with halting conditions
w1 = perceptronclassifier(shufdata,w,N); % wrong here!

% decide final W
W = finalweight(shufdata,w1,N);

% compute error rate with final W
j = 0;
for i=1:N
    if (dot(W,shufdata(i,:))<=0)
        j = j+1;
    end
end
etrain = j/N;

% classify test data
N1 = size(feature_test,1);
D1 = size(feature_test,2);
testdata = ones(N1,D1+1);
testdata(:,2)=feature_test(:,1);
testdata(:,3)=feature_test(:,2);
etest = errorate(testdata,label_test,N1,W);

% plot boundray
plotDecBoundaries(feature_train,label_train,W);
