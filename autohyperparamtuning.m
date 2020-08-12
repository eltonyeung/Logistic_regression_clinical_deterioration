%Predicting Clincial Deterioration
%% README
% This script is the main script you should run 
%   - Architecture of this logistic regression is adapted from MINST Image 
%     classification code by Blake Richard (blake.richards@mila.quebec)
                    
% (1) The code should run error free and output updates.
% (2) The neural network should begin training and output the loss and error values.
% (3) The synaptic weight matrix image should be generated, as well as plots of the loss and error over epochs.

%                 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
%                 !!!!!!!!!!!Prerequisites!!!!!!!!!!!!
%                 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
% Make sure to meet the following pre-reqs for the model to run properly.
% 1) Have the following functions located in the same folder.
%        "clinicaldeterioration.m", "load_data.m", "calculate_error.m", "calculate_loss.m", 
%        "sigmoid.m", "forward_pass.m", and "backwardpass.m".
% 2) Edit the source of clinical score table on "load_data.m" for importing data
% 3) Choose the label of interest to be predicted by editing the index of
%    feature of interest in "load_data.m"


% STEPS:
% 1) Step 1: Initilising the hyperparameters, weights, and load data in
%            desired format.
% 2) Step 2: Choose a cross-validation to train with: K-fold, or Leave-one-out cross-validation 
%            (the current ideal one is LOOCV, scroll down to see the LOOCV option)
% 3) Step 3: Test the latest trained model (weights and bias) on the
%            original dataset
% 4) Step 4: AFTER TESTING, the parameters for the trained model will be
%            saved in the structure called 'trainedmodel', you can save the
%            trained model by doing this.

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% STEP 1 - Initializing %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% load the data
if exist('train_features') ~= 1
  [selectedfeatures,selectedlabels, ntrainsubjects, ntestsubjects, score, all_labels, all_features] = load_data();
end
%options: [train_features, train_abels, test_features, test_labels,
%selectedfeatures, ntrainsubjects, ntestsubjects, score, all_labels, all_features]

%%%%%%%%%%%%%%%%%%%% INITIALIZE HYPERPARAMETERS
p_epsilon = [0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05];   % the learning rate (initial = 0.01)
p_alpha   = [0.003, 0.005];  % the momentum (initial = 0.001)
p_gamma   = [0.0005, 0.00075, 0.0015, 0.0020]; % the weight decay (initial = 0.0001)
p_nhid = [5, 6, 7, 8, 9];  % the number of hidden units
sigma   = 0.1;    % the variance in the initial weights and biases
p_nepoch  = [75,100,125,150,175,200,225,250];    % the number of training epochs 
%p_nepoch  = [100, 150, 200, 250];
%initialise counter
counter = 1;

for iepsilon = 1:length(p_epsilon)
    epsilon = p_epsilon(iepsilon);
    for ialpha = 1:length(p_alpha)
        alpha = p_alpha(ialpha);      
        for igamma = 1:length(p_gamma)
            gamma = p_gamma(igamma);
            for inhid = 1:length(p_nhid)
                nhid = p_nhid(inhid);
                for inepoch = 1:length(p_nepoch)
                    nepoch = p_nepoch(inepoch);
                    
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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% STEP 2 - Training %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%- TRAIN THE NEURAL NETWORK USING LEAVE-ONE-OUT CROSS VALIDATION
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%initialize batch size and #epoch 
nbatch  = length(all_features);   % the number of mini matches for training 
batch_size = nbatch-1;

% print a message
fprintf('|||-----------------------------------------------------------------\n');
fprintf('Beginning neural network training:\n');
fprintf('|||-----------------------------------------------------------------\n');

for batch = 1:nbatch
    % Split dataset into the training batch and the LOO-subject for testing
    leaveout_features = all_features(:,batch);  %leave one out subject
    leaveout_labels = all_labels(:,batch); %leave one out subject
       
    batch_features = all_features(:,setdiff(1:nbatch, batch)); %read subject features and leaving one out 
    batch_labels = all_labels(:,setdiff(1:nbatch, batch)); %read subject labels and leaving one out

%     %%%%%%%%%%%%%% Oversample the minority class %%%%%%%%%%%%%%%%%
%     minority = (batch_labels(2,:));  %%%%% Manually set the minority group (label 2)
%     minority_features= batch_features(:,minority == 1);
%     minority_labels= batch_labels(:,minority ==1);
%    
%     %concatenate minority cluster into all for oversampling
%     batch_features = [batch_features, minority_features];
%     batch_labels = [batch_labels, minority_labels];
%         
%     balance(batch).deteriorated = sum(batch_labels(2,:));
%     balance(batch).nodeterioration = length(batch_features) - balance(batch).deteriorated;
%     
    %print msg
    fprintf(':::::::::::::::::::::::Batch %2.0f ::::::::::::::::::::::::: \n',batch);
    
    %%%%%%%%%%%%%%TRAIN the MODEL USING THIS BATCH FOR nEPOCHS
     for epoch = 1:nepoch  
        % perform a forward pass
        [r0,r1] = forward_pass(batch_features,W0,W1,b0,b1);
        
        % calculate the loss and error on this batch
        train_loss(epoch) = calculate_loss(r1,batch_labels);
        train_accuracy(epoch) = 1-calculate_error(r1,batch_labels);  %NEWLY ADDED for tracking train_error
        
        %%%%%%%%%%%%%%%%% Update the weight and bias 
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
              
        % print a message
        fprintf('Epoch %d, training loss = %2.3f, training accuracy = %2.1f%%.\n',epoch,train_loss(epoch),100.0*train_accuracy(epoch)); 
    end 


    %%%%%%%%%%%%%%%%% Update the batch result
    %record the average batch loss and batch training accuracy
    batch_loss(batch,:) = train_loss;
    batch_accuracy(batch,:) = train_accuracy;
    
    %print msg for showing this batch
    fprintf('*****Batch %d, Batch loss = %2.3f, Batch accuracy = %2.1f%%.\n',batch, mean(batch_loss(batch,:)), 100*mean(batch_accuracy(batch,:)));   
    
    %%%%%%%%%%%%%%%%% Test on LOO subject using the trained model 
    [test_r0,test_r1] = forward_pass(leaveout_features,W0,W1,b0,b1); 
    [E, classification] = calculate_error(test_r1,leaveout_labels);
    LOO_error(batch) = E;
    LOO_output(:,batch) = test_r1;
    LOO_labels(:,batch) = leaveout_labels;
    LOO_classification(:,batch) = classification;
    
    %%%%%%%%%%%%%%%%%EDIT: Reinitialise weights and bias for new batch
    % Reinitialize the synaptic weights and biases
    W0 = randn(nhid,numel(selectedfeatures)); % weights from input to hidden layer
    W1 = randn(2,nhid);  % weights from hidden layer to output
    b0 = randn(nhid,1);   % biases at hidden layer
    b1 = randn(2,1);     % biases at output layer

    % Reinitialize the weight and bias changes
    delta_W0 = zeros(nhid,numel(selectedfeatures)); 
    delta_W1 = zeros(2,nhid);  
    delta_b0 = zeros(nhid,1);   
    delta_b1 = zeros(2,1);     
     
end 

%%%%%%%% Calculate TOTAL AVERAGE loss and accuracy across all the training and LOO tests 
LOO_loss = calculate_loss(LOO_output, LOO_labels);
LOO_accuracy = 100* (1- mean(LOO_error));
train_loss = mean(batch_loss,1);
train_accuracy = 100* mean(batch_accuracy,1);

cc = confusionmat(LOO_classification(2,:), LOO_labels(2,:));
sensitivity = cc(2,2)/(cc(2,2)+cc(1,2));   %TP / TP+FN
specificity = cc(1,1)/(cc(1,1)+cc(2,1));   %TN / TN+FP
F1 = 2*(  ((cc(2,2)/(cc(1,2)+cc(2,2))) * (cc(2,2)/(cc(2,1)+cc(2,2))))/((cc(2,2)/(cc(1,2)+cc(2,2))) + (cc(2,2)/(cc(2,1)+cc(2,2)))))

% plot the loss curve
% figure(1);
% plot([1:nepoch],train_loss,'k-','LineWidth',1);
% box off;
% xlabel('Epoch');
% ylabel('Loss (AU)');
% set(gca,'FontSize',10);
% set(gcf,'numbertitle','off','name','Training loss');
% title('Training Loss');
% %print('loss.png','-dpng','-r300');
% 
% % plot the error curve
% figure(2);
% plot([1:nepoch],train_accuracy,'r-','LineWidth',1);
% box off;
% xlabel('Epoch');
% ylabel('Accuracy (%)');
% set(gca,'FontSize',10);
% set(gcf,'numbertitle','off','name','Training accuracy');
% title('Training accuracy');
% %print('error.png','-dpng','-r300');
% 
% figure(3);
% cm =confusionchart(LOO_classification(2,:), LOO_labels(2,:));
% cm.YLabel = 'Predicted Class';
% cm.XLabel = 'Actual Class';
% cm.Title = 'Leave-one-out Classification';
% cm.RowSummary = 'row-normalized';
% cm.ColumnSummary = 'column-normalized';
% set(gcf,'numbertitle','off','name','Cumulative Leave-one-out Classification');

% print a message
fprintf('|||-----------------------------------------------------------------\n');
fprintf('Finished training. Training loss = %2.3f, accuracy = %2.1f%%.\n',train_loss(end),train_accuracy(end));
fprintf('                   LOOCV loss = %2.3f, accuracy = %2.1f%%.\n',LOO_loss,LOO_accuracy);
fprintf('>>>Hyperparameters: nhid = %d, nepoch = %d.\n',nhid, nepoch);
fprintf('                    epsilon = %2.3f, alpha = %2.3f, gamma = %2.4f, sigma = %2.2f.\n',epsilon,alpha,gamma,sigma);
fprintf('|||-----------------------------------------------------------------\n');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% STEP 3 - SAVE THE MODEL %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Save all the parameters used in this run


trainedmodel(counter).selectedfeatures = selectedfeatures;	
trainedmodel(counter).W0 = W0;
trainedmodel(counter).W1 = W1;
trainedmodel(counter).b0 = b0;
trainedmodel(counter).b1 = b1;
trainedmodel(counter).nepoch = nepoch;
trainedmodel(counter).nbatch = nbatch;
trainedmodel(counter).nhid = nhid;
trainedmodel(counter).epsilon = epsilon;
trainedmodel(counter).alpha = alpha;
trainedmodel(counter).gamma = gamma;
trainedmodel(counter).all_labels = all_labels;
trainedmodel(counter).all_features = all_features;
trainedmodel(counter).train_accuracy = batch_accuracy;
trainedmodel(counter).train_loss = batch_loss;
trainedmodel(counter).LOO_output = LOO_output;
trainedmodel(counter).LOO_loss = LOO_loss;
trainedmodel(counter).LOO_classification = LOO_classification;
trainedmodel(counter).LOO_error = LOO_error;
trainedmodel(counter).LOO_accuracy = LOO_accuracy;
trainedmodel(counter).Sensitivity = sensitivity;
trainedmodel(counter).Specificity = specificity;
trainedmodel(counter).F1score = F1;

%progress tracker
counter = counter + 1


clear train_loss train_accuracy W0 W1 b0 b1 r0 r1 balance batch_loss batch_accuracy batch_features batch_labels classification delta* dL* E LOO* minority* test_r0 test_r1 

                end
            end
        end
    end
end


           
%%
%Find the best model
hypertune = struct2table(trainedmodel);

%best accuracy
idx(hypertune.LOO_accuracy == max(hypertune.LOO_accuracy)) = 1;
rownum = find(idx,1,'last');
model.acc = hypertune(rownum,:);
 fprintf('Best accuracy %2.3f\n',max(hypertune.LOO_accuracy)); 
 

%best f1
idx2(hypertune.F1score == max(hypertune.F1score)) = 1;
rownum2 = find(idx2,1,'last');
model.f1=hypertune(rownum2,:);
 fprintf('Best F1-score %2.3f\n',max(hypertune.F1score)); 
 
 
%best specificity
idx3(hypertune.Specificity == max(hypertune.Specificity)) = 1;
rownum3 = find(idx3,1,'last');
model.specificity=hypertune(rownum3,:);
 fprintf('Best specificity %2.3f\n',max(hypertune.Specificity)); 


%best sensitivity
idx4(hypertune.Sensitivity == max(hypertune.Sensitivity)) = 1;
rownum4 = find(idx4,1,'last');
model.sensitivity=hypertune(rownum4,:);
 fprintf('Best sensitivity %2.3f\n',max(hypertune.Sensitivity)); 

%%
% plot the loss curve
figure(1);
plot([1:trainedmodel(selected).nepoch],mean(trainedmodel(selected).train_loss,1),'k-','LineWidth',1);
box off;
xlabel('Epoch');
ylabel('Loss (AU)');
set(gca,'FontSize',10);
set(gcf,'numbertitle','off','name','Training loss','Position',[75,75,375,300]);
title('Training Loss');
%print('loss.png','-dpng','-r300');


% plot the error curve
figure(2);
plot([1:trainedmodel(selected).nepoch],mean(trainedmodel(selected).train_accuracy,1),'r-','LineWidth',1);
box off;
xlabel('Epoch');
ylabel('Accuracy (%)');
set(gca,'FontSize',10);
set(gcf,'numbertitle','off','name','Training accuracy','Position',[75,75,375,300]);
title('Training accuracy');
%print('error.png','-dpng','-r300');

% plot the LOOCV CM
figure(3);
cm =confusionchart(trainedmodel(selected).LOO_classification(2,:), trainedmodel(selected).all_labels(2,:));
cm.YLabel = 'Predicted Class';
cm.XLabel = 'Actual Class';
cm.Title = 'Leave-one-out Classification';
set(gcf,'numbertitle','off','name','Cumulative Leave-one-out Classification','Position',[75,75,375,300]);




