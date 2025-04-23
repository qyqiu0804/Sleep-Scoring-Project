% Step 1: Load EDF and XML files
addpath('Scripts');

% Load data
datadir = 'Data';
edfFiles = dir([datadir '/*.edf']);
edfFiles = edfFiles(~startsWith({edfFiles.name}, '~$'));
xmlFiles = dir([datadir '/*.xml']);
xmlFiles = xmlFiles(~startsWith({xmlFiles.name}, '~$'));
band_freqs = [0.6,1; 0.5,4; 4,8; 8,13; 11,16; 13,30];

num = length(edfFiles);
[X, Y] = compile_classification_data(datadir, edfFiles, xmlFiles, band_freqs, 1:num);

% Standardize features
X = normalize(X);

% Step 2: Create the loss function (return total loss, not average)
myfun = @(Xtrain, ytrain, ~, ~) ...
    kfoldLoss(crossval(fitcknn(Xtrain, ytrain, ...
    'NumNeighbors', 12, ...
    'Distance', 'cityblock', ...
    'DistanceWeight', 'squaredinverse', ...
    'Standardize', true), ...
    'KFold', 10));

% Step 3: Stratified 10-fold cross-validation
cv = cvpartition(Y, 'KFold', 10);

% Step 4: Feature selection with sequentialfs
opts = statset('Display', 'iter');
[selected, history] = sequentialfs(myfun, X, Y, ...
    'cv', cv, ...
    'options', opts);

% Display results
disp('Selected features (column indices):');
disp(find(selected));

% Reduce to selected features
X_selected = X(:, selected);

% Train final model with selected features
finalModel = fitcknn(X_selected, Y, ...
    'NumNeighbors', 12, ...
    'Distance', 'cityblock', ...
    'DistanceWeight', 'squaredinverse', ...
    'Standardize', true);

% Step 5: Cross-validation performance
cv_final = crossval(finalModel);
classError = kfoldLoss(cv_final)

% Confusion matrix
labels = {'N1','N2','N3','N4','REM','Wake'};
Y_pred = kfoldPredict(cv_final);
conf_matrix = confusionmat(Y, Y_pred);
conf_matrix_percent = conf_matrix ./ sum(conf_matrix, 2) * 100;

% Plot it
figure;
imagesc(conf_matrix_percent);
xlabel('Predicted'); ylabel('True');
xticklabels(labels); yticklabels(labels);
title('Confusion Matrix Percentages (After Corrected Feature Selection)');
colorbar;
