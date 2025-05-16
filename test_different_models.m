% Configuration
modelType = 'rf';         % Choose: 'svm', 'rf', 'nn', 'knn'
cvStrategy = 'loso';       % Choose: 'kfold', 'loso'
rng(1);                    % For reproducibility

% Load Data
addpath('Scripts');

% Load data
datadir = 'Data';
edfFiles = dir([datadir '/*.edf']);
edfFiles = edfFiles(~startsWith({edfFiles.name}, '~$'));
xmlFiles = dir([datadir '/*.xml']);
xmlFiles = xmlFiles(~startsWith({xmlFiles.name}, '~$'));
band_freqs = [0.6,1; 0.5,4; 4,8; 8,13; 11,16; 13,30];

num = length(edfFiles);
[X, Y, subject_bounds] = compile_classification_data(datadir, edfFiles, xmlFiles, band_freqs, 1:num);

% Apply selected features (computed through the file "select_features")
selected = [1 2 3 4 10 11 12 14 19 24 56 57 59 63 69 72 75 84 86];
X_selected = X(:, selected);

% Setup
labels = {'REM','N3','N2','N1','Wake'};
unique_classes = unique(Y);
nClasses = numel(unique_classes);
Y_pred = zeros(size(Y));

% Cross-validation
if strcmpi(cvStrategy, 'kfold')
    cv = cvpartition(Y, 'KFold', 5);
    for i = 1:cv.NumTestSets
        trainIdx = training(cv, i);
        testIdx = test(cv, i);
        model = train_model(X_selected(trainIdx,:), Y(trainIdx), modelType);
        Y_pred(testIdx) = predict_model(model, X_selected(testIdx,:), modelType);
    end
elseif strcmpi(cvStrategy, 'loso')
    startIdx = 1;
    for s = 1:length(subject_bounds)
        endIdx = subject_bounds(s);
        testIdx = startIdx:endIdx;
        trainIdx = setdiff(1:size(X_selected,1), testIdx);
        model = train_model(X_selected(trainIdx,:), Y(trainIdx), modelType);
        Y_pred(testIdx) = predict_model(model, X_selected(testIdx,:), modelType);
        startIdx = endIdx + 1;
    end
else
    error('Unknown CV strategy.');
end

% Metrics
overallAcc = mean(Y_pred == Y) * 100;
conf_matrix = confusionmat(Y, Y_pred, 'Order', unique_classes);
conf_matrix_percent = conf_matrix ./ sum(conf_matrix, 2) * 100;
perClassAcc = diag(conf_matrix) ./ sum(conf_matrix, 2);
f1_scores = compute_f1(conf_matrix);

fprintf('\n=== Performance Report (%s | %s) ===\n', upper(modelType), upper(cvStrategy));
fprintf('Overall Accuracy: %.2f%%\n', overallAcc);
for i = 1:nClasses
    fprintf('Class %s - Accuracy: %.2f%%, F1-score: %.2f\n', ...
        labels{i}, perClassAcc(i)*100, f1_scores(i));
end

% Confusion Matrix Plot
figure;
imagesc(conf_matrix_percent);
axis equal tight;
set(gca, 'XTick', 1:nClasses, 'XTickLabel', labels);
set(gca, 'YTick', 1:nClasses, 'YTickLabel', labels);
xlabel('Predicted'); ylabel('True');
title(sprintf('Confusion Matrix (%s | %s)', upper(modelType), upper(cvStrategy)));
colorbar;

% ROC Curve (per class)
% Make sure model gives scores / probabilities
if strcmpi(modelType, 'rf')
    [~, scores_all] = predict(model, X_selected);  % Final model used on all data
elseif strcmpi(modelType, 'svm')
    [~, scores_all] = predict(model, X_selected);
elseif strcmpi(modelType, 'nn')
    [~, scores_all] = predict(model, X_selected);
elseif strcmpi(modelType, 'knn')
    [~, scores_all] = predict(model, X_selected);
else
    error('ROC plotting not supported for this model.');
end

% Convert ground truth to numeric index format
[~, ~, Y_numeric] = unique(Y);  % e.g., 1 to 5
Y_onehot = zeros(length(Y), nClasses);
for i = 1:length(Y)
    Y_onehot(i, Y_numeric(i)) = 1;
end

% Plot ROC curves
figure;
hold on;
colors = lines(nClasses);
aucs = zeros(nClasses,1);

for i = 1:nClasses
    [Xroc, Yroc, ~, AUC] = perfcurve(Y_onehot(:, i), scores_all(:, i), 1);
    plot(Xroc, Yroc, 'Color', colors(i,:), 'LineWidth', 2);
    aucs(i) = AUC;
end

legend(arrayfun(@(i) sprintf('%s (AUC = %.2f)', labels{i}, aucs(i)), 1:nClasses, 'UniformOutput', false), ...
    'Location', 'SouthEast');
xlabel('False Positive Rate');
ylabel('True Positive Rate');
title(sprintf('ROC Curves (%s | %s)', upper(modelType), upper(cvStrategy)));
grid on;
hold off;



% Helper Functions
function model = train_model(X, Y, modelType)
    switch lower(modelType)
        case 'svm'
            t = templateSVM('KernelFunction', 'rbf', ...
                            'KernelScale', 5.9503, ...
                            'BoxConstraint', 192.9);
            model = fitcecoc(X, Y, 'Learners', t, 'Coding', 'onevsall');
        case 'rf'
            model = TreeBagger(100, X, Y, 'Method', 'classification');
        case 'nn'
            model = fitcnet(X, Y);
        case 'knn'
            model = fitcknn(X, Y, ...
                'NumNeighbors', 12, ...
                'Distance', 'cityblock', ...
                'DistanceWeight', 'squaredinverse', ...
                'Standardize', true);
        otherwise
            error('Unsupported model type.');
    end
end

function Yp = predict_model(model, X, modelType)
    if strcmp(modelType, 'rf')
        Yp = str2double(predict(model, X));
    else
        Yp = predict(model, X);
    end
end

function f1 = compute_f1(conf)
    TP = diag(conf);
    FP = sum(conf,1)' - TP;
    FN = sum(conf,2) - TP;
    f1 = 2*TP ./ (2*TP + FP + FN);
    f1(isnan(f1)) = 0;
end
