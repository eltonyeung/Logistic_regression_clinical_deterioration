function [selectedfeatures, selectedlabels, ntrainsubjects, ntestsubjects, score, all_labels, all_features] = load_data()
%[TRAIN_IMAGES, TRAIN_LABELS, TEST_IMAGES, TEST_LABELS] = LOAD_DATA() loads the training and testing data
% 	global ntrainsubjects ntestsubjects nsubjects selectedfeatures features ntrainsubjects ntestsubjects nsubjects score
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%% - INITIALISING - %%%%%%%%%%%%%%%%%%%%%%%%%%
% open the files for reading
score = readtable('D:\Box Sync\Box Sync\Clinical Scores\Elton Clinical Deterioration\finalResult_Percentagetime(UnannotatedApple)V4-pb_EYupdate(DayTimeOnly).xlsx','Range','A1:AV37','TreatAsEmpty','true');

%%%%%%EDIT 1: put together column 13 and 14, combining the pre-calculated ranking scores
score.Var13(isnan(score.Var13)) = score.Var14(isnan(score.Var13)); 

%%%%%%EDIT 2: Merging posture into 4 classes: Bed, Chair, Stand, Walk
score.Ppos1 = score.Ppos1 + score.Ppos0;
score.Properties.VariableNames{9} = 'Merged_Ppos1';

% %%%%%%%%%%%% mRS Edit 1: Preadmission Binorizing 
% score.mRS_pre(score.mRS_pre <= 1) =0; %class 1 = mRS (0 , 1)
% score.mRS_pre(score.mRS_pre >= 2) =1; %class 2 = mRS (2 , 3)
% % remove nan rows
% score(isnan(score.mRS_pre),:) = [];

% %%%%%%%%%%%% mRS Edit 2: Postadmission Binorizing 
% score.mRS_post(score.mRS_post <= 2) =0; %class 1 = mRS (0 , 1)
% score.mRS_post(score.mRS_post >= 3) =1; %class 2 = mRS (2 , 3)
% % % remove nan rows
% score(isnan(score.mRS_post),:) = [];
%   
%%%%%%%%%%%%%% NIHSS Edit 1: Summing across NIHSS motor score
% score.NIHSSmotor_admission = sum([score.NIHSS_motor_Larm, score.nNIHSS_motor_Rarm, score.NIHSS_motor_Lleg, score.NIHSS_motor_Rleg],2)
% %binorisation [class 1 = 0,    class 2 = 1+]
% score.NIHSSmotor_admission(score.NIHSSmotor_admission >= 1) =1; %class 2 = NIHSS >1
% score(isnan(score.NIHSSmotor_admission),:) = [];

%extract features for training the model with
features = score.Properties.VariableNames;
%8-12 = %time spent in Pos1 to Pos5

%%%%%%%%%%%%%%%%%%%%%%%%%%% - SELECT FEATURES HERE %%%%%%%%%%%%%%%%%%%%%%%
selectedfeatures = {features{[4,9:12]}};   %pin point the features we want

%%%%%%%%%%%%%%%%%%%%%%%%%%% - SELECT LABELS HERE %%%%%%%%%%%%%%%%%%%%%%%
selectedlabels = {features{21}};  %classification: deterioration (1/0)

%21 = comp_pb (deterioration (1/0))
%47 = mRS_pre (mRS 0-1 or 2+)
%48 = mRS_post (mRS 0-2, 3+)
%49 = NIHSSmotor

%%%%%%%%%%% - INITIALIZE PARAMETERS FOR READING DATA - K-FOLD VALIDATION
%%%%%%%%%%% - Start of K-Fold validation code
%pin the subjects to train the model with (subjects that were annotated by groundtruth raters)
% test_subjects = {'SG9042';'SG9043';'SG9056';'SG9055';'SG9023';'SG9046'};  %mRS, 3 * ones and 3 * zeros

test_subjects = {'SG9049'}; %COMP_PB
% test_subjects = {'SG9049';'SG9045';'SG9038';'SG9039';'SG9055';'SG9046'}; %COMP_PB
%initialize parameters
ntestsubjects = numel(test_subjects);
nsubjects = height(score);
ntrainsubjects = nsubjects - ntestsubjects;

% locate row numbers for testing subjects
list = table2array(score(:,2));
for i = 1:ntestsubjects
     idx = contains(list,test_subjects(i));
     rownumtest(i) = find(idx,1,'last');
end 

% locate row numbers for training subjects
rownumtrain =[1:nsubjects]';
rownumtrain(rownumtest) = [];

% initialize the data holders
train_features = zeros(ntrainsubjects,numel(selectedfeatures));
train_labels = zeros(ntrainsubjects,2);  %labels = deterioration (Y/N)
test_features = zeros(ntestsubjects,numel(selectedfeatures));
test_labels = zeros(ntestsubjects,2);  %labels = deterioration (Y/N)


%%%%%%%%%%% - READ IN THE TRAINING FEATURES AND LABELS
for i = 1:ntrainsubjects
    for k = 1:numel(selectedfeatures)
        train_features(i,k) = score.(string({selectedfeatures(k)}))(rownumtrain(i));
    end
    
%     for j =1:numel(selectedlabels)
%         train_labels(i,j) = score.(string({selectedlabels(j)}))(rownumtrain(i));  
%     end 

    for j =1:numel(selectedlabels)
         if score.(string({selectedlabels(j)}))(rownumtrain(i)) == 0,
            train_labels(i,1) = 1;
        else train_labels(i,2) = 1;
         end
    end 
    
end

%%%%%%%%%%% - READ IN THE TEST FEATURES AND LABELS
% read the testing features and labels
for i = 1:numel(rownumtest)
    for k = 1:numel(selectedfeatures)
        test_features(i,k) = score.(string({selectedfeatures(k)}))(rownumtest(i));
    end
    
%     for j =1:numel(selectedlabels)
%         test_labels(i,j) = score.(string({selectedlabels(j)}))(rownumtest(i));  
%     end 
    for j =1:numel(selectedlabels)
         if score.(string({selectedlabels(j)}))(rownumtest(i)) == 0,
            test_labels(i,1) = 1;
        else test_labels(i,2) = 1;
         end
    end 
    
end

%Transpose them for feeding into network
test_features = test_features';
test_labels = test_labels';
train_features = train_features';
train_labels = train_labels';

%Normalise train and test features
train_features = (train_features - mean(train_features,2,'omitnan'))./std(train_features,[],2,'omitnan');
test_features = (test_features - mean(test_features,2,'omitnan'))./std(test_features,[],2,'omitnan');

%imputate data mean for missing values
train_features(isnan(train_features) ==1) = 0;
test_features(isnan(test_features) ==1) = 0;
%%%%%%%%%%%%%% END OF K-FOLD VALIDATION DATA READING %%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%% - Preparation for LEAVE-ONE-OUT VALIDATION ANLAYSES
%group them into ALL for leave-one-out analyses
all_features = [test_features , train_features];
all_labels = [test_labels , train_labels];

%%%%%%%%%%%%% - Normalize data by feature 
scaled_features = (all_features - mean(all_features,2,'omitnan'))./std(all_features,[],2,'omitnan');
all_features = scaled_features;

%imputate data mean for missing values
all_features(isnan(all_features) ==1) = 0;

%make a copy of the original dataset for final testing
% final_features = all_features;
% final_labels = all_labels;


%%%%%%%%%%%%% - OVERSAMPLE the deterioration cluster '1's (minority) by doubleing them
%               (from n=12 out of 36 to n=24 / 48) 
% minority(all_labels(2,:) == 1) = 1;
% minority_features= all_features(:,minority == 1);
% minority_labels= all_labels(:,minority ==1);
% 
% %concatenate minority cluster into all for oversampling
% all_features = [all_features, minority_features];
% all_labels = [all_labels, minority_labels];


% %%%%%%%NEW: identify and replicate minority group (1)
% minority(train_labels(2,:) == 1) = 1;
% minority_features= train_features(:,minority == 1);
% minority_labels= train_labels(:,minority ==1);
% 
% %concatenate minority cluster into training set 
% train_features = [train_features, minority_features];
% train_labels = [train_labels, minority_labels];



% %%%%%%%%%%%%%% - OVERSAMPLING + UNDERSAMPLING FOR mRS
% %oversample minority class
% minority(train_labels(2,:) == 1) = 1;
% minority_features= train_features(:,minority == 1);
% minority_labels= train_labels(:,minority ==1);
% 
% %undersample majority class
% majority(train_labels(2,:) == 0) = 1;
% majority_features= train_features(:,majority == 1);
% majority_labels = train_labels(:,majority == 1);
% majority_features(:,15:21) = [];
% majority_labels(:,15:21) = [];
% 
% %concatenate minority cluster into all for oversampling
% train_features = [minority_features, majority_features, minority_features];
% train_labels = [minority_labels, majority_labels, minority_labels];
% 



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% END OF LOOCV PREPARATION %%%%%%%%%%%%%%%%%%%%

% function end
end
