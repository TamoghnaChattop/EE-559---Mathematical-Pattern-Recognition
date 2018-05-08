function y = errorate(dataset,label,datanum,weight)
j = 0;
for i=1:datanum
    if ((dot(weight,dataset(i,:))<=0)&&(label(i,:)==1))
        j = j+1;
    elseif ((dot(weight,dataset(i,:))>0)&&(label(i,:)==2))
        j = j+1;
    end
end
y = j/datanum;