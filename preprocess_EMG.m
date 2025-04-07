function filtered_signal = preprocess_EMG(EMG_signal, fs)
    % Preprocesses EMG signal by applying high-pass filtering and RMS smoothing
    %inline with standards for EMG data

    % Step 1: High-pass Filter at 20 Hz
    [high_b, high_a] = butter(4, 20 / (fs / 2), 'high');
    EMG_highpassed = filtfilt(high_b, high_a, EMG_signal);

    % Step 2: RMS Filter using a 100 ms window
    window_samples = round(fs * 0.1);  % 100 ms
    filtered_signal = sqrt(movmean(EMG_highpassed.^2, window_samples));

   
end
