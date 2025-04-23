% Step 1: Load EDF and XML files
addpath('Scripts');

datadir = 'Data';
edfFiles = dir([datadir '/*.edf']);
edfFiles = edfFiles(~startsWith({edfFiles.name}, '~$'));
xmlFiles = dir([datadir '/*.xml']);
xmlFiles = xmlFiles(~startsWith({xmlFiles.name}, '~$'));
band_freqs = [0.6,1; 0.5,4; 4,8; 8,13; 11,16; 13,30];

num = length(edfFiles);
[X, Y] = compile_classification_data(datadir, edfFiles, xmlFiles, band_freqs, 1:num);

% Step 2: Standardize before PCA
X = normalize(X);

% Step 3: Run PCA
[coeff, score, ~, ~, explained, mu] = pca(X);

% Try different numbers of components
num_components_list = [10, 20, 30, 40, 50];
errors = zeros(size(num_components_list));

% Loop through each setting
for i = 1:length(num_components_list)
    n = num_components_list(i);

    % Project to n-dimensional PCA space
    X_reduced = score(:, 1:n);

    % Train model on reduced features
    model = fitcknn(X_reduced, Y, ...
        'NumNeighbors', 12, ...
        'Distance', 'cityblock', ...
        'DistanceWeight', 'squaredinverse', ...
        'Standardize', true);

    % Cross-validation
    cv_model = crossval(model, 'KFold', 10);
    classError = kfoldLoss(cv_model);
    errors(i) = classError;

    fprintf('PCA with %d components -> Classification error: %.4f\n', n, classError);
end

% Step 4: Plot results
figure;
plot(num_components_list, errors, '-o', 'LineWidth', 2);
xlabel('Number of Principal Components');
ylabel('Classification Error (10-fold CV)');
title('PCA Feature Reduction Performance');
grid on;
