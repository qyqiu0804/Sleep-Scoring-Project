function feature_matrix = extract_time_frequency_features(filtered_signal, fs, winLength, stepLength)
    % Convert window/step length to samples
    winSamples = round(winLength * fs);  % Window size in samples
    stepSamples = round(stepLength * fs); % Step size in samples

    % Compute number of epochs (moving windows)
    num_epochs = floor((length(filtered_signal) - winSamples) / stepSamples) + 1;

    % Preallocate feature matrix
    feature_matrix = zeros(num_epochs, 13); % 13 features per window

    % Precompute indices for sliding windows
    indices = arrayfun(@(i) (i-1)*stepSamples + (1:winSamples), 1:num_epochs, 'UniformOutput', false);
    indices = cell2mat(indices');

    % Filter signals once for K-complex, Spindles, and Slow Waves
    bandpass_filter_order = 4;
    [bandpass_b, bandpass_a] = butter(bandpass_filter_order, [0.3 3] / (0.5 * fs), 'bandpass');
    low_filtered_signal = filtfilt(bandpass_b, bandpass_a, filtered_signal);
    spindle_band = bandpass(filtered_signal, [11 16], fs);
    spindle_envelope = abs(hilbert(spindle_band));
    sw_band = bandpass(filtered_signal, [0.5 2], fs);

    % Extract all epochs at once
    epoch_signals = filtered_signal(indices);
    epoch_low_signals = low_filtered_signal(indices);
    epoch_spindle_envelopes = spindle_envelope(indices);
    epoch_sw_bands = sw_band(indices);

    % Disable warnings for findpeaks
    warning('off', 'signal:findpeaks:largeMinPeakHeight');

    % Mean, Variance, Skewness, Kurtosis, ZCR (calculated per epoch)
    mean_vals = mean(epoch_signals, 2);  % Mean across columns (for each epoch)
    var_vals = var(epoch_signals, 0, 2);  % Variance across columns (for each epoch)
    skew_vals = skewness(epoch_signals, 0, 2);  % Skewness across columns (for each epoch)
    kurt_vals = kurtosis(epoch_signals, 0, 2);  % Kurtosis across columns (for each epoch)
    
    % ZCR calculation: Zero Crossing Rate (sum of zero-crossings per epoch)
    zcr_vals = sum(abs(diff(sign(epoch_signals), 1, 2)), 2) / winSamples;  % Zero-crossing rate

    % Precompute K-Complex, Spindles, and Slow Waves for all epochs
    k_threshold = 75;
    k_locs = arrayfun(@(i) length(findpeaks(epoch_low_signals(:, i), 'MinPeakHeight', k_threshold, 'MinPeakDistance', fs * 0.5)), 1:num_epochs);

    spindle_thresholds = mean(epoch_spindle_envelopes) + 2 * std(epoch_spindle_envelopes);
    spindle_locs = arrayfun(@(i) length(findpeaks(epoch_spindle_envelopes(:, i), 'MinPeakHeight', spindle_thresholds(i), 'MinPeakDistance', fs * 0.5)), 1:num_epochs);

    sw_thresholds = mean(epoch_sw_bands) - 2 * std(epoch_sw_bands);
    sw_locs = arrayfun(@(i) length(findpeaks(-epoch_sw_bands(:, i), 'MinPeakHeight', sw_thresholds(i), 'MinPeakDistance', fs * 1)), 1:num_epochs);

    sw_percentages = (sw_locs / winSamples) * 100;

    % Compute Hjorth parameters
    hjorth_params_matrix = zeros(num_epochs, 3);  % Preallocate the matrix with 3 columns for each epoch
    for i = 1:num_epochs
        [activity, mobility, complexity] = hjorth_params(epoch_signals(i, :));  % Get Hjorth parameters for each epoch
        hjorth_params_matrix(i, :) = [activity, mobility, complexity];  % Store them in the matrix
    end
    % Store features in matrix
    feature_matrix = [mean_vals, var_vals, skew_vals, kurt_vals, zcr_vals, hjorth_params_matrix, k_locs', spindle_locs', sw_locs', sw_percentages'];

    % Re-enable warnings
    warning('on', 'signal:findpeaks:largeMinPeakHeight');
end
