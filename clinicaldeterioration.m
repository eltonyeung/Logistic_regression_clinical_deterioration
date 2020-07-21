%Predicting Clincial Deterioration

% This script is the main script you should run to test your work and write-up your report. If you have done
% your code properly, then the following should happen when you type 'nrod08main' into the Matlab prompt:
%
% (1) The code should run error free and output updates.
% (2) The neural network should begin training and output the loss and error values.
% (3) The synaptic weight matrix image should be generated, as well as plots of the loss and error over epochs.

clc 
clear 

% load the data
if exist('train_features') ~= 1
    [train_features, train_labels, test_features, test_labels, selectedfeatures] = load_data()
end

% initialize various hyperparameters
epsilon = 0.01;   % the learning rate
alpha   = 0.001;  % the momentum
gamma   = 0.0001; % the weight decay
nepoch  = 1000;    % the number of training epochs
nbatch  = 5;   % the number of mini matches for training
% nhid    = size(selectedfeatures,2);    % the number of hidden units
nhid = 8
sigma   = 0.1;    % the variance in the initial weights and biases

% initialize the synaptic weights and biases
W0 = randn(nhid,numel(selectedfeatures)); % weights from input to hidden layer
W1 = randn(2,nhid);  % weights from hidden layer to output
b0 = randn(nhid,1);   % biases at hidden layer
b1 = randn(2,1);     % biases at output layer

% initialize the weight and bias changes
delta_W0 = zeros(nhid,numel(selectedfeatures)); 
delta_W1 = zeros(2,nhid);  
delta_b0 = zeros(nhid,1);   
delta_b1 = zeros(2,1);     

% initialize the batch image and label holders
batch_size   = length(train_features)/nbatch;
batch_features = zeros(numel(selectedfeatures),batch_size);
batch_labels = zeros(2,batch_size);

% initialize the loss and error holders
batch_loss = zeros(nbatch,1);
train_loss = zeros(nepoch+1,1);
test_error = zeros(nepoch+1,1);

%%
% print a message
fprintf('|||-----------------------------------------------------------------\n');
fprintf('Beginning neural network training:\n');
fprintf('|||-----------------------------------------------------------------\n');

% do a pre-training test of the loss and error rate
[train_r0,train_r1] = forward_pass(train_features,W0,W1,b0,b1); 
[test_r0,test_r1]   = forward_pass(test_features,W0,W1,b0,b1); 
train_loss(1)       = calculate_loss(train_r1,train_labels)/nbatch;
test_error(1)       = calculate_error(test_r1,test_labels);
fprintf('Pre-training loss = %2.3f, test error = %2.1f%%.\n',train_loss(1),100.0*test_error(1));

%%%%%%% - TRAIN THE NEURAL NETWORK
% for each epoch...
for epoch = 1:nepoch

	% for each batch
	for batch = 1:nbatch

		% set the images and labels
		batch_features = train_features(:,(batch-1)*batch_size+1:batch*batch_size); 
		batch_labels = train_labels(:,(batch-1)*batch_size+1:batch*batch_size); 

		% perform a forward pass
		[r0,r1] = forward_pass(batch_features,W0,W1,b0,b1);

		% perform a backward pass
		[dL_by_dW0,dL_by_dW1,dL_by_db0,dL_by_db1] = backward_pass(batch_features,batch_labels,r0,r1,W0,W1,b0,b1);

		% calculate the parameter updates
		delta_W0 = -epsilon*dL_by_dW0 + alpha*delta_W0 - gamma*W0; 
		delta_W1 = -epsilon*dL_by_dW1 + alpha*delta_W1 - gamma*W1; 
		delta_b0 = -epsilon*dL_by_db0 + alpha*delta_b0 - gamma*b0; 
		delta_b1 = -epsilon*dL_by_db1 + alpha*delta_b1 - gamma*b1; 

		% update the parameters
		W0 = W0 + delta_W0;	
		W1 = W1 + delta_W1;	
		b0 = b0 + delta_b0;	
		b1 = b1 + delta_b1;	

		% calculate the loss on this batch
		batch_loss(batch) = calculate_loss(r1,batch_labels);
	end

	% calculate the average loss and test error
	train_loss(epoch+1) = mean(batch_loss);
	[test_r0,test_r1] = forward_pass(test_features,W0,W1,b0,b1); 
	test_error(epoch+1) = calculate_error(test_r1,test_labels);
	
	% print a message
	fprintf('Epoch %d, loss = %2.3f, test error = %2.1f%%.\n',epoch,train_loss(epoch+1),100.0*test_error(epoch+1));

end

% show the final weights
%show_weights(W0);
%print('weights.png','-dpng','-r300');

% plot the loss curve
figure(2);
plot([0:nepoch],train_loss,'k-','LineWidth',1);
box off;
xlabel('Epoch');
ylabel('Loss (AU)');
set(gca,'FontSize',10);
set(gcf,'numbertitle','off','name','Train loss');
%print('loss.png','-dpng','-r300');

% plot the error curve
figure(3);
plot([0:nepoch],test_error*100.0,'r-','LineWidth',1);
box off;
xlabel('Epoch');
ylabel('Error (%)');
set(gca,'FontSize',10);
set(gcf,'numbertitle','off','name','Test error');
%print('error.png','-dpng','-r300');

% print a message
fprintf('|||-----------------------------------------------------------------\n');
fprintf('Finished training. Figures have been saved in the current folder.\n');
fprintf('|||-----------------------------------------------------------------\n');


%%
%% Custom stuff

  %opens classification learner app
 % classificationLearner
  %%
  
    yfit = trainedModel3.predictFcn(svmtrain);
    cm = confusionchart(train_labels,yfit)
 
%%
 yfit = trainedModel.predictFcn(svmtrain);
 
 RMSE = sqrt(mean((yfit - train_labels).^2)); 
 
 scatter(train_labels,yfit);
 xlabel('true score');
 ylabel('predicted score');
 ylim([0 12]);

%% SVM
%prepare matrix for SVM
    svmtrain = [train_features, train_labels];
    svmtest = [test_features, test_labels];

    %opens classification learner app
%   classificationLearner
  
  regressionLearner
  
yfit = trainedModel.predictFcn(test_features)

%% MLR
train_mlr_model = fitrlinear(train_labels,[train_features]');

%Execute Prediction
train_mlr_pred= predict(train_mlr_model,X);

%Correlation of predicted function
train_mlr_corr = corr(train_mlr_pred,Y);
rmse_mlr_train= sqrt(mean(abs(train_mlr_pred - score_wellbeing_train).^2));

%Figure output
figure();
hold on
scatter(Y,train_mlr_pred,'filled')
xlabel('WB score observed')
ylabel('WB score predicted')
title('Multiple Linear Regression Prediction (Training)')
T = (strcat('RMSE =',{' ' },num2str(rmse_mlr_train)));
T2 = (strcat('r =',{' ' },num2str(train_mlr_corr)));
text(-1,2.5,T, 'FontSize',10);
text(-1,2,T2, 'FontSize',10);


%% 
%%  %%  %% scoring and ranking method
  
 testscore = [];
 trainscore = [];
  
  for i = 1:size(test_features,1)
    testscore(i) = (test_features(i,5) + 2* test_features(i,6) + 3* test_features(i,7) + 4* test_features(i,8) + 5* test_features(i,9));
  end  
  
  for i = 1:size(train_features,1)
    trainscore(i) = (train_features(i,5) + 2* train_features(i,6) + 3* train_features(i,7) + 4* train_features(i,8) + 5* train_features(i,9));
  end 
  
    train_features = [train_features, trainscore'];
    test_features = [test_features, testscore'];
        
    selectedfeatures = [selectedfeatures, {'rankingscore'}];

