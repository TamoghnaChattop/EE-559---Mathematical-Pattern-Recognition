clc;
clear all;

%Load the data
load('D:\python3_files\python3\synthetic1.mat');

%Size of data points
row = ((size(feature_train,1)));
col = ((size(feature_train,2)));

%Augmented data
Augdata = ones(row,col+1);
Augdata(:,2) = feature_train(:,1);
Augdata(:,3) = feature_train(:,2);

%Reverse the data points
for i = 1:row
    if(label_train(i) == 2)
        Augdata(i,:) = Augdata(i,:)*(-1);
    end
end

%Randomize the training data
augdata = Augdata(randperm(row),:);

%Define weight vector
w = 0.1*ones(row,col+1);

%Define counter for ad-hoc condition 1
count = 0;

%Define cost function
J = zeros(100,1);

%Classifier Loop
for j = 0:1000
    for i = 1:row-1
        if (w(i,1)*augdata(i,1)+w(i,2)*augdata(i,2)+w(i,3)*augdata(i,3) <= 0)
            w(i+1,:) = w(i,:)+augdata(i,:);
        else 
            w(i+1,:) = w(i,:);
            count = count+1;
        end
    end
    if (count == row)
        break
    end
    w(1,:) = w(i+1,:);
    
    %Second adhoc condition
    if (j==1000)
        for k = 1:row
            for i = 1:row
                if (w(k,1)*augdata(i,1)+w(k,2)*augdata(i,2)+w(k,3)*augdata(i,3) <= 0)
                    J(k) = J(k) - w(k,1)*augdata(i,1)-w(k,2)*augdata(i,2)-w(k,3)*augdata(i,3);
                end
            end
            if (J(k)==0)
                J(k) = 11111;
            end
        end
        [~,index] = min(J);
    end
end

%Final Weight Vector
weight = w(index,:);

%Compute train error rate with final W
count2 = 0;
for i = 1:row
     if (weight(1,1)*augdata(i,1)+weight(1,2)*augdata(i,2)+weight(1,3)*augdata(i,3) <= 0)
        count2=count2+1;
     end
end
train_error = count2/row;

%Plot Decision Boundray
plotDecBoundaries(feature_train,label_train,weight);

 
