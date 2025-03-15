function eeg_features = extract_features_eeg(EEG_filtered_signal, fs, band_freqs)

    % Extract basic features using Short-Time Fourier Transform (STFT)
    [spec_snr, entropy, specBinWidthHz, freqs] = extract_features_spectrogram(EEG_filtered_signal, fs, 30, 99);
    entropy = smooth(entropy, 21)'; % Smooth entropy to reduce local fluctuations

    % Compute band power ratios
    band_ratios = compute_band_power_ratios(spec_snr, specBinWidthHz, band_freqs);

    % Identify peak frequencies
    [~, peak_freq_inds] = max(spec_snr);
    peak_freqs = freqs(peak_freq_inds)';

    % Compute band entropies
    band_entropies = compute_band_entropies(spec_snr, specBinWidthHz, band_freqs, freqs);
    for k2 = 1:6
        band_entropies(k2, :) = smooth(band_entropies(k2, :), 20);
    end

    % Extract time-frequency features
    time_freq_features = extract_time_frequency_features(EEG_filtered_signal, fs, 99, 30);

% AR Model Feature Extraction
ar_features = extract_ar_features(EEG_filtered_signal, fs);

% Pad ar_features to match other feature dimensions (if needed)
target_len = size(time_freq_features', 2); 

if length(ar_features) ~= target_len
    mean_value = nanmean(ar_features);  % Use mean value
    ar_features = padarray(ar_features, [0, target_len - length(ar_features)], mean_value, 'post');
end

% Combine extracted features into a single output
eeg_features = [band_ratios; entropy; peak_freqs; band_entropies; time_freq_features'; ar_features];
