function eeg_features = extract_features_eeg(EEG_filtered_signal, fs, band_freqs)

    % ===== Step 1: Extract basic features using Short-Time Fourier Transform (STFT) =====
    [spec_snr, entropy, specBinWidthHz, freqs] = extract_features_spectrogram(EEG_filtered_signal, fs, 30, 99);
    entropy = smooth(entropy, 21)';

    band_ratios = compute_band_power_ratios(spec_snr, specBinWidthHz, band_freqs);

    [~, peak_freq_inds] = max(spec_snr);
    peak_freqs = freqs(peak_freq_inds)';

    band_entropies = compute_band_entropies(spec_snr, specBinWidthHz, band_freqs, freqs);
    for k2 = 1:6
        band_entropies(k2, :) = smooth(band_entropies(k2, :), 20);
    end

    time_freq_features = extract_time_frequency_features(EEG_filtered_signal, fs, 99, 30);

    % ===== Step 2: Extract AR features for each frame =====
    num_frames = size(time_freq_features, 1);
    window_len_samples = round(2 * fs);
    step_samples = round(0.25 * window_len_samples);

    ar_feature_list = [];

    for i = 1:num_frames
        start_idx = (i - 1) * step_samples + 1;
        end_idx = start_idx + window_len_samples - 1;
        if end_idx > length(EEG_filtered_signal)
            break;
        end
        eeg_seg = EEG_filtered_signal(start_idx:end_idx);
        ar_feat = extract_ar_features(eeg_seg, fs);
        ar_feature_list = [ar_feature_list; ar_feat];
    end

    ar_features = ar_feature_list;
    num_valid_frames = size(ar_features, 1);

    % ===== Step 3: Combine all features frame by frame =====
    eeg_features = [];
    for i = 1:num_valid_frames
        f = [ ...
            band_ratios(:, i)', ...
            entropy(:, i), ...
            peak_freqs(:, i), ...
            band_entropies(:, i)', ...
            time_freq_features(i, :), ...
            ar_features(i, :)
        ];
        eeg_features = [eeg_features; f];
    end
% === Transpose to [features × frames] format for later use ===
eeg_features = eeg_features';  % [features × frames]

% === Fix feature dimension to a consistent value ===
fixed_feature_dim = 58;  % or whatever your standard is

if size(eeg_features, 1) < fixed_feature_dim
    eeg_features(end+1:fixed_feature_dim, :) = NaN;
elseif size(eeg_features, 1) > fixed_feature_dim
    eeg_features = eeg_features(1:fixed_feature_dim, :);
end
end
