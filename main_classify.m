% Step 1: Load EDF and XML files
addpath('Scripts'); % Set the folder path containing the scripts

% Specify the path to your EDF and XML files
datadir = 'Data';
edfFiles = dir([datadir filesep '*.edf']); 
xmlFiles = dir([datadir filesep '*.xml']);

% specify the frequencies of each wave type to use in the functions below
band_freqs = [0.6,1; 0.5,4; 4,8;  8,13; 11,16; 13,30];

num = length(edfFiles);

%% training
% go over the training portion, and make big X and Y vectors
[X, Y] = compile_classification_data(datadir, edfFiles, xmlFiles,band_freqs,1:num);

model = fitcknn(X,Y,'NumNeighbors',22,'Standardize',1);

rng(1); % For reproducibility

%% check k-fold loss
crossv = crossval(model);
classError = kfoldLoss(crossv)

%% confusion matrix
Y_pred = kfoldPredict(crossv);
conf_matrix = confusionmat(Y,Y_pred);
conf_matrix_percent = conf_matrix./sum(conf_matrix,2) * 100 %put it in percents instead
imagesc(conf_matrix_percent); 
xlabel('predicted'); ylabel('truth');
