basePath = 'D:/Che Qiancao/KTH/Course/12/Signal/Lab/';
addpath(fullfile(basePath, 'LabCode'));

dataFolder = fullfile(basePath, 'Project Data');
edfFilename = fullfile(dataFolder, 'R1.edf');
fs = 125;
channel = 8;

% Load EEG signal
[hdr, record] = edfread(edfFilename);
EEG_signal = record(channel, :);

% Preprocess EEG signal
preprocessed_EEG_signal = preprocess_EEG(EEG_signal, fs);

% Define bands
filter_order = 4;
delta_band = [0.5 4];
theta_band = [4 8];
alpha_band = [8 13];
beta_band = [13 30];

% Apply Butterworth bandpass filters directly in main script
[b_delta, a_delta] = butter(filter_order, delta_band / (fs/2), 'bandpass');
delta_signal = filtfilt(b_delta, a_delta, preprocessed_EEG_signal);

[b_theta, a_theta] = butter(filter_order, theta_band / (fs/2), 'bandpass');
theta_signal = filtfilt(b_theta, a_theta, preprocessed_EEG_signal);

[b_alpha, a_alpha] = butter(filter_order, alpha_band / (fs/2), 'bandpass');
alpha_signal = filtfilt(b_alpha, a_alpha, preprocessed_EEG_signal);

[b_beta, a_beta] = butter(filter_order, beta_band / (fs/2), 'bandpass');
beta_signal = filtfilt(b_beta, a_beta, preprocessed_EEG_signal);

% Compute AR model, PSD, and select best order
max_order = 30;
Nfft = 1024;
[ar_coeffs.delta, PSD_delta, F_delta, best_order] = compute_ar_psd(delta_signal, fs, max_order, Nfft);
[ar_coeffs.theta, PSD_theta, F_theta, ~] = compute_ar_psd(theta_signal, fs, max_order, Nfft);
[ar_coeffs.alpha, PSD_alpha, F_alpha, ~] = compute_ar_psd(alpha_signal, fs, max_order, Nfft);
[ar_coeffs.beta, PSD_beta, F_beta, ~] = compute_ar_psd(beta_signal, fs, max_order, Nfft);

% Compute band ratios
ratios = compute_band_ratios(PSD_delta, PSD_theta, PSD_alpha, PSD_beta, F_delta, F_theta, F_alpha, F_beta);

% Find peak frequencies
peak_freqs.delta = find_peak_frequency(F_delta, PSD_delta, delta_band);
peak_freqs.theta = find_peak_frequency(F_theta, PSD_theta, theta_band);
peak_freqs.alpha = find_peak_frequency(F_alpha, PSD_alpha, alpha_band);
peak_freqs.beta = find_peak_frequency(F_beta, PSD_beta, beta_band);

%% === Display Results ===
fprintf('\n==== AR Model Features Extracted ====\n');

% AR Model Coefficients
fprintf('\nAR Model Coefficients:\n');
fprintf('Delta Band: %s\n', mat2str(ar_coeffs.delta, 4));
fprintf('Theta Band: %s\n', mat2str(ar_coeffs.theta, 4));
fprintf('Alpha Band: %s\n', mat2str(ar_coeffs.alpha, 4));
fprintf('Beta Band: %s\n', mat2str(ar_coeffs.beta, 4));

% Spectral Peak Frequencies
fprintf('\nSpectral Peak Frequencies (Hz):\n');
fprintf('Delta: %.2f Hz\n', peak_freqs.delta);
fprintf('Theta: %.2f Hz\n', peak_freqs.theta);
fprintf('Alpha: %.2f Hz\n', peak_freqs.alpha);
fprintf('Beta: %.2f Hz\n', peak_freqs.beta);

% Spectral Band Ratios
fprintf('\nSpectral Band Ratios:\n');
fprintf('Delta Ratio: %.4f\n', ratios.delta);
fprintf('Theta Ratio: %.4f\n', ratios.theta);
fprintf('Alpha Ratio: %.4f\n', ratios.alpha);
fprintf('Beta Ratio: %.4f\n', ratios.beta);
