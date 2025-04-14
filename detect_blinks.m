function blink_info = detect_blinks(signal, fs)
    % Filter
    filtered = medfilt1(signal, 5);
    [b, a] = butter(2, [0.5 4] / (fs / 2), 'bandpass');
    filtered = filtfilt(b, a, filtered);

    % Energy envelope
    energy = filtered .^ 2;
    energy = smoothdata(energy, 'movmean', round(0.2 * fs));
    min_height = 0.5 * max(energy);
    [pks, locs] = findpeaks(energy, 'MinPeakHeight', min_height);

    candidate_times = locs / fs;
    merged_times = [];

    if ~isempty(candidate_times)
        merged_times = candidate_times(1);
        for i = 2:length(candidate_times)
            if candidate_times(i) - merged_times(end) >= 1.0
                merged_times(end+1) = candidate_times(i); %#ok<AGROW>
            end
        end
    end

    % Initialize output
    blink_info.times = merged_times;
    blink_info.durations = zeros(size(merged_times));
    blink_info.amplitudes = zeros(size(merged_times));
    blink_info.IBI = diff(merged_times);

    % Estimate duration and amplitude per blink
    for i = 1:length(merged_times)
        idx = round(merged_times(i) * fs);
        window = round(0.5 * fs);  % +/- 0.5 s window
        left = max(1, idx - window);
        right = min(length(energy), idx + window);
        local_energy = energy(left:right);

        % Duration: width over 30% of peak energy
        e_peak = energy(idx);
        threshold = 0.3 * e_peak;
        above_thresh = find(local_energy >= threshold);
        if ~isempty(above_thresh)
            duration = (above_thresh(end) - above_thresh(1)) / fs;
        else
            duration = 0.1; % fallback
        end

        blink_info.durations(i) = duration;

        % Amplitude: from raw (unfiltered) signal
        sig_window = signal(left:right);
        blink_info.amplitudes(i) = max(sig_window) - min(sig_window);
    end
end
