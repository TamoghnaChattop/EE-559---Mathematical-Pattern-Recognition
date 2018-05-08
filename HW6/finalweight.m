function y = finalweight(shufdata,weight,datanum)
J=zeros(datanum,1);
for j=1:datanum
    for i=1:datanum
        if (dot(weight(j,:),shufdata(i,:))<=0)
            J(j) = J(j)-dot(weight(j,:),shufdata(i,:));
        end
    end
    if (J(j)==0)
        J(j) = 10000;
    end
end
[~,index] = min(J);
y = weight(index,:);
