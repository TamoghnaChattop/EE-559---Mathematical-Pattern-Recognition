% 1 vs rest classification: mean1 vs mean23,mean2 vs mean13,mean3 vs Mean1(2)

%compute sample mean
load('wine.mat');
j=1;
k=1;
r=1;
s=size(feature_train,1);
feature_train1=zeros(s,2);
feature_train2=zeros(s,2);
feature_train3=zeros(s,2);
for i=1:s
    if (label_train(i)==1)
        feature_train1(j,:)=feature_train(i,1:2);
        j=j+1;
    elseif(label_train(i)==2)
        feature_train2(k,:)=feature_train(i,1:2);
        k=k+1;
    else
        feature_train3(r,:)=feature_train(i,1:2);
        r=r+1;
    end
end
Mean1 = sum(feature_train1,1)./(j-1);%class1 mean
Mean2 = sum(feature_train2,1)./(k-1); %class2 mean
Mean3 = sum(feature_train3,1)./(r-1); %class3 mean
Mean12 = (sum(feature_train1,1)+sum(feature_train2,1))./(s-r+1);
Mean13 = (sum(feature_train1,1)+sum(feature_train3,1))./(s-k+1);
Mean23 = (sum(feature_train2,1)+sum(feature_train3,1))./(s-j+1);

%classification accuracy on training&test dataset
a=0;
for i=1:s
    dis1=sqrt((feature_train(i,1)-Mean1(1))^2+(feature_train(i,2)-Mean1(2))^2);
    dis2=sqrt((feature_train(i,1)-Mean2(1))^2+(feature_train(i,2)-Mean2(2))^2);
    dis3=sqrt((feature_train(i,1)-Mean3(1))^2+(feature_train(i,2)-Mean3(2))^2);
    dis12=sqrt((feature_train(i,1)-Mean12(1))^2+(feature_train(i,2)-Mean12(2))^2);
    dis13=sqrt((feature_train(i,1)-Mean13(1))^2+(feature_train(i,2)-Mean13(2))^2);
    dis23=sqrt((feature_train(i,1)-Mean23(1))^2+(feature_train(i,2)-Mean23(2))^2);
    if (dis1<dis23)&&(dis13<dis2)&&(dis12<dis3)&&(label_train(i)==1)
        a=a+1;
    end
    if (dis2<dis13)&&(dis23<dis1)&&(dis12<dis3)&&(label_train(i)==2)
        a=a+1;
    end
    if (dis3<dis12)&&(dis13<dis2)&&(dis23<dis1)&&(label_train(i)==3)
        a=a+1;
    end
end
accutrain = a./s;

a1=0;
s1=size(feature_test,1);
for i=1:s1
    dis1=sqrt((feature_test(i,1)-Mean1(1))^2+(feature_test(i,2)-Mean1(2))^2);
    dis2=sqrt((feature_test(i,1)-Mean2(1))^2+(feature_test(i,2)-Mean2(2))^2);
    dis3=sqrt((feature_test(i,1)-Mean3(1))^2+(feature_test(i,2)-Mean3(2))^2);
    dis12=sqrt((feature_test(i,1)-Mean12(1))^2+(feature_test(i,2)-Mean12(2))^2);
    dis13=sqrt((feature_test(i,1)-Mean13(1))^2+(feature_test(i,2)-Mean13(2))^2);
    dis23=sqrt((feature_test(i,1)-Mean23(1))^2+(feature_test(i,2)-Mean23(2))^2);
    if (dis1<dis23)&&(dis13<dis2)&&(dis12<dis3)&&(label_test(i)==1)
        a1=a1+1;
    elseif (dis2<dis13)&&(dis23<dis1)&&(dis12<dis3)&&(label_test(i)==2)
        a1=a1+1;
    elseif (dis3<dis12)&&(dis13<dis2)&&(dis23<dis1)&&(label_test(i)==3)
        a1=a1+1;
    end
end
accutest = a1./s;

%plot 2-class decision boundraies
sample_mean1=[Mean1;Mean23];
sample_mean2=[Mean2;Mean13];
sample_mean3=[Mean3;Mean12];
plotDecBoundaries2(feature_train, label_train, sample_mean1);
plotDecBoundaries2(feature_train, label_train, sample_mean2);
plotDecBoundaries2(feature_train, label_train, sample_mean3);

%plot final decision boundraies and regions
plotDecBoundaries3(feature_train, label_train, sample_mean1,sample_mean2,sample_mean3);
