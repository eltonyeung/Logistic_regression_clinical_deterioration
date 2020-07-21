function [train_features, train_labels, test_features, test_labels, selectedfeatures, ntrainsubjects, ntestsubjects, score] = load_data()
%[TRAIN_IMAGES, TRAIN_LABELS, TEST_IMAGES, TEST_LABELS] = LOAD_DATA() loads the training and testing data
%	
global ntrainsubjects ntestsubjects nsubjects selectedfeatures features ntrainsubjects ntestsubjects nsubjects score

% open the files for reading
score = readtable('D:\Box Sync\Box Sync\Clinical Scores\Elton Clinical Deterioration\finalResult_Percentagetime(UnannotatedApple)V4-pb.xlsx','Range','A1:AT37','TreatAsEmpty','true');


%pin the subjects to train the model with (subjects that were annotated by groundtruth raters)
test_subjects = {'SG9049';'SG9045';'SG9038';'SG9039';'SG9055';'SG9046'}   %1/6 of 1s and 0s

% {'SG9004';'SG9025';'SG9027';'SG9028';'SG9029';'SG9033'};
%{'SG9007'; 'SG9035'; 'SG9043'; 'SG9045'; 'SG9052'; 'SG9054'; 'SG9055'};

%extract features for training the model with
features = score.Properties.VariableNames;
%8-12 = %time spent in Pos1 to Pos5


%%%%%% - SELECT FEATURES HERE
selectedfeatures = {features{[4,5,8:12]}};   %pin point the features we want

%%%%%% - SELECT LABELS HERE
selectedlabels = {features{21}};  %classification: deterioration (1/0)
% selectedlabels = {features{30}};  %regression: NIHSS upon admission


%%%%%% - INITIALIZE PARAMETERS FOR READING DATA
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


%%%%%% - READ IN THE TRAINING IMAGES AND LABELS
for i = 1:ntrainsubjects
    for k = 1:numel(selectedfeatures)
        train_features(i,k) = score.(string({selectedfeatures(k)}))(rownumtrain(i));
    end
    
%     for j =1:numel(selectedlabels)
%         train_labels(i,j) = score.(string({selectedlabels(j)}))(rownumtrain(i));  
%     end 

    for j =1:numel(selectedlabels)
         if score.(string({selectedlabels(j)}))(rownumtrain(i)) == 0,
            train_labels(i,1) = 1
        else train_labels(i,2) = 1;
         end
    end 
    
end

%%%%%% - READ IN THE TEST IMAGES AND LABELS
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
            test_labels(i,1) = 1
        else test_labels(i,2) = 1;
         end
    end 
    
end

% TRANSPOSE MX FOR NEXT STEPS
train_features = train_features';
test_features = test_features';
train_labels = train_labels';
test_labels = test_labels';

% function end
end
