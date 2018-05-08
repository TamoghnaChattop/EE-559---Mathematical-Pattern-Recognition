function [] = plotDecBoundaries3(training, label_train, sample_mean1,sample_mean2,sample_mean3)
%Plot the decision boundaries and data points for minimum distance to
%class mean classifier

% training: traning data
% label_train: class lables correspond to training data
% sample_mean: mean vector for each class

% Total number of classes
nclass =  max(unique(label_train));
nclasss = 2;
 
% Set the feature range for ploting
max_x = ceil(max(training(:, 1))) + 1;
min_x = floor(min(training(:, 1))) - 1;
max_y = ceil(max(training(:, 2))) + 1;
min_y = floor(min(training(:, 2))) - 1;

xrange = [min_x max_x];
yrange = [min_y max_y];

% step size for how finely you want to visualize the decision boundary.
inc = 0.005;

% generate grid coordinates. this will be the basis of the decision
% boundary visualization.
[x, y] = meshgrid(xrange(1):inc:xrange(2), yrange(1):inc:yrange(2));

% size of the (x, y) image, which will also be the size of the
% decision boundary image that is used as the plot background.
image_size = size(x);                                         %question!
xy = [x(:) y(:)]; % make (x,y) pairs as a bunch of row vectors.

% distance measure evaluations for each (x,y) pair.
dist_mat1 = pdist2(xy, sample_mean1); %dist_mat1 is x*y by 2.
[~, pred_label1] = min(dist_mat1, [], 2); %pred_lable1 is x*y by 1, value is 1 or 2
dist_mat2 = pdist2(xy, sample_mean2);
[~, pred_label2] = min(dist_mat2, [], 2);
dist_mat3 = pdist2(xy, sample_mean3);
[~, pred_label3] = min(dist_mat3, [], 2); 

pred_label=zeros(size(pred_label1,1),1);
for i=1:size(pred_label1,1)
    if (pred_label1(i,:)==1)&&(pred_label2(i,:)==2)&&(pred_label3(i,:)==2)
        pred_label(i,:)=1;
    elseif (pred_label1(i,:)==2)&&(pred_label2(i,:)==1)&&(pred_label3(i,:)==2)
        pred_label(i,:)=2;
    elseif (pred_label1(i,:)==2)&&(pred_label2(i,:)==2)&&(pred_label3(i,:)==1)
        pred_label(i,:)=3;
    else
        pred_label(i,:)=4;
    end
end
% reshape the idx (which contains the class label) into an image.
decisionmap = reshape(pred_label, image_size);

figure;
 
%show the image, give each coordinate a color according to its class label
imagesc(xrange,yrange,decisionmap);
hold on;
set(gca,'ydir','normal');
 
% colormap for the classes:
cmap = [1 0.8 0.8; 0.95 1 0.95; 0.9 0.9 1;0.5 0.5 0.5];
colormap(cmap);
 
% plot the class training data.
plot(training(label_train == 1,1),training(label_train == 1,2), 'rx');
plot(training(label_train == 2,1),training(label_train == 2,2),'go');
if nclass == 3
    plot(training(label_train == 3,1),training(label_train == 3,2), 'b*');
end

% include legend for training data
if nclass == 3
    legend('Class 1', 'Class 2', 'Class 3', ...
    'Location','northoutside','Orientation', 'horizontal');
else
    legend('Class 1', 'Class 2', ...
    'Location','northoutside','Orientation', 'horizontal');
end

% plot the class mean vector.
% mean1 = plot(sample_mean(1,1),sample_mean(1,2), 'rd', ...
%              'MarkerSize', 8, 'MarkerFaceColor', 'r');
% mean2 = plot(sample_mean(2,1),sample_mean(2,2), 'gd', ...
%              'MarkerSize', 8, 'MarkerFaceColor', 'g');
% if nclasss == 3
%     mean3 = plot(sample_mean(3,1),sample_mean(3,2), 'bd',...
%                 'MarkerSize', 8, 'MarkerFaceColor', 'b');
% end

% create a new axis for lengends of class mean vectors
ah=axes('position',get(gca,'position'),'visible','off');

% include legend for class mean vector
% if nclasss == 3
%     legend(ah, [mean1, mean2, mean3], {'Class 1 Mean', 'Class 2 Mean', 'Class 3 Mean'}, ...
%     'Location','northoutside','Orientation', 'horizontal');
% else
%     legend(ah, [mean1, mean2], {'NewClass 1 Mean', 'NewClass 2 Mean'},  ...
%     'Location','northoutside','Orientation', 'horizontal');
% end

% label the axes.
xlabel('featureX');
ylabel('featureY');
end
