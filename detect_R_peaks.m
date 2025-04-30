function R_peaks = detect_R_peaks(ecg_signal, fs)
% detect_R_peaks - Stable and adaptive R peak detection for long ECG recordings
% Inputs:
%   ecg_signal - preprocessed ECG signal (e.g., after wavelet denoising)
%   fs - sampling frequency (Hz)
% Output:
%   R_peaks - indices of detected R peaks (sample indices)

% Divide the signal into segments (e.g., every 60 seconds)
segment_duration = 60; % duration of each segment in seconds
samples_per_segment = segment_duration * fs;
n_segments = ceil(length(ecg_signal) / samples_per_segment);

R_peaks = [];

for i = 1:n_segments
    idx_start = (i-1)*samples_per_segment + 1;
    idx_end = min(i*samples_per_segment, length(ecg_signal));
    segment = ecg_signal(idx_start:idx_end);

    % 1. Automatically determine R wave polarity (positive or negative)
    if abs(min(segment)) > abs(max(segment))
        segment_to_detect = -segment; % R waves are negative
    else
        segment_to_detect = segment;  % R waves are positive
    end

    % 2. Smooth the signal using moving average
    window_size = round(0.15 * fs);
    smoothed = movmean(segment_to_detect, window_size);

    % 3. Dynamic local thresholding based on each segment
    local_threshold = 0.1 * max(smoothed);

    % 4. Preliminary peak detection
    [~, locs] = findpeaks(smoothed, ...
        'MinPeakHeight', local_threshold, ...
        'MinPeakDistance', round(0.5 * fs)); % Minimum distance of 0.5 seconds between peaks

    % 5. Fine adjustment: locate the true maximum/minimum in the original signal
    refined_peaks = zeros(size(locs));
    search_window = round(0.4 * fs); % Search within ±0.4 seconds
    for j = 1:length(locs)
        search_start = max(locs(j) + idx_start - 1 - search_window, 1);
        search_end = min(locs(j) + idx_start - 1 + search_window, length(ecg_signal));
        if abs(min(segment)) > abs(max(segment))
            % If R waves are negative, find the local minimum
            [~, idx] = min(ecg_signal(search_start:search_end));
        else
            % If R waves are positive, find the local maximum
            [~, idx] = max(ecg_signal(search_start:search_end));
        end
        refined_peaks(j) = search_start + idx - 1;
    end

    R_peaks = [R_peaks refined_peaks]; %#ok<AGROW>
end

% --- Add abnormal amplitude filtering here ---
% Remove abnormally large peaks (likely artifacts)

peak_amplitudes = ecg_signal(R_peaks);
amplitude_median = median(abs(peak_amplitudes), 'omitnan');
amplitude_threshold = 2.5 * amplitude_median; % Threshold multiplier (adjustable)

valid_idx = abs(peak_amplitudes) < amplitude_threshold;
R_peaks = R_peaks(valid_idx);

% Sort the final detected R peaks
R_peaks = sort(R_peaks);

end
