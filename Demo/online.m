%% online LGPR
function [updateTimeOnline,preOnline,errOLGP]=online(onlineModel,ave,x_train,y_train,x_test,y_test)
for i=1:size(x_train,2)-ave
    onlineModel.updatediv(x_train(:,i), y_train(:,i));
end
tic;
for i=size(x_train,2)-ave+1:size(x_train,2)
    onlineModel.updatediv(x_train(:,i), y_train(:,i));
end
updateTimeOnline = toc./ave;
%disp(updateTimeOnline(k));
% predict weighted average
ypred_prob = zeros(1,size(x_test,2));
tic;
for i=1:size(x_test,2)
    ypred_prob(:,i) = onlineModel.prediv(x_test(:,i));
end
preOnline=toc./size(x_test,2);
% error
errOLGP = sum(mean((ypred_prob - y_test).^2));
end