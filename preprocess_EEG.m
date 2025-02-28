function filtered_signal = preprocess_EEG(EEG_signal, fs)
    % Preprocesses EEG signal by performing baseline correction, notch filtering, and bandpass filtering

    % Step 1: Baseline Correction (Remove DC Offset)
    baseline_segment = EEG_signal(1:500); % First 500 points as baseline
    baseline_mean = mean(baseline_segment);
    EEG_signal_baseline_corrected = EEG_signal - baseline_mean;

    % Step 2: Notch Filter (48-52 Hz) to remove power line noise (EU standard)
    notch_filter_order = 2; % Filter order
    notch_low = 48; % Lower cutoff for notch
    notch_high = 52; % Upper cutoff for notch
    [notch_b, notch_a] = butter(notch_filter_order, [notch_low, notch_high] / (0.5 * fs), 'stop');
    EEG_signal_notch_filtered = filtfilt(notch_b, notch_a, EEG_signal_baseline_corrected);

    % Step 3: Bandpass Filter (0.3-35 Hz) to keep relevant EEG frequencies
    bandpass_filter_order = 4; % Filter order
    low_cutoff = 0.3; % Lower cutoff
    high_cutoff = 35; % Upper cutoff
    [bandpass_b, bandpass_a] = butter(bandpass_filter_order, [low_cutoff, high_cutoff] / (0.5 * fs), 'bandpass');
    filtered_signal = filtfilt(bandpass_b, bandpass_a, EEG_signal_notch_filtered);

    % later: add in artifact removal
    % filtered_signal = eeg_remove_artifacts(filtered_signal, events);
end
