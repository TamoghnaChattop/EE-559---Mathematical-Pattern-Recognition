function y = perceptronclassifier(shufdata,weight,datanum)
N = size(diff(weight),1);
D = size(diff(weight),2);
for j=0:999
    for i=1:datanum
        if (dot(weight(i,:),shufdata(i,:)) <= 0)
            weight(i+1,:) = weight(i,:)+shufdata(i,:);
        else
            weight(i+1,:) = weight(i,:);
        end
    end
    y = weight(1:datanum,:);
    if (diff(y) == zeros(N,D))
        break;
    end
    weight(1,:) = weight(i+1,:);
end

%readme
%weight(i) ~= weight(i,:)
