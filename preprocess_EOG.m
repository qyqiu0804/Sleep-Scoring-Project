function filtered_signal = preprocess_EOG(EOG_signal, fs)
    %highpass filter for baseline correction
    %2nd order butterworth
    %using 0.3 Hz based on canvas paper
    fc = 0.3; % Cutoff frequency (Hz)
    [b, a] = butter(2, fc / (fs / 2), 'high'); % 2nd-order Butterworth filter
    filtered_signal = filtfilt(b, a, EOG_signal);
end