function filtered_signal = preprocess_ECG(ECG_signal, fs)
    % Baseline correction
    ECG_baseline_corrected = detrend(ECG_signal);
    
    %Notch filter for EU powerline noise
    [notch_b, notch_a] = butter(2, [48, 52] / (0.5 * fs), 'stop');
    ECG_notch_filtered = filtfilt(notch_b, notch_a, ECG_baseline_corrected);

    %bandpass filter
    [bandpass_b, bandpass_a] = butter(2, [0.5, 45] / (0.5 * fs), 'bandpass');
    filtered_signal = filtfilt(bandpass_b, bandpass_a, ECG_notch_filtered);
end