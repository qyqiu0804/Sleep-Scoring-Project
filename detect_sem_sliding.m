function [sem_times, sem_indices, sem_pattern] = detect_sem_sliding(loc, roc, fs, window_sec, overlap_ratio)
% Strict SEM detection based on standard flowchart with square + moving avg
% Also returns pattern info: ISI, burst count, burst size, density

    arguments
        loc (1,:) double
        roc (1,:) double
        fs (1,1) double
        window_sec (1,1) double = 2
        overlap_ratio (1,1) double = 0.75
    end

    % Design filters
    band1Filt = designfilt('bandpassiir','FilterOrder',4, ...
        'HalfPowerFrequency1',0.5,'HalfPowerFrequency2',6, ...
        'SampleRate',fs);
    band2Filt = designfilt('bandpassiir','FilterOrder',4, ...
        'HalfPowerFrequency1',1,'HalfPowerFrequency2',6, ...
        'SampleRate',fs);
    band3Filt = designfilt('bandpassiir','FilterOrder',4, ...
        'HalfPowerFrequency1',1.5,'HalfPowerFrequency2',6, ...
        'SampleRate',fs);

    % Sliding window
    L = length(loc);
    window_len = round(window_sec * fs);
    step_len = round(window_len * (1 - overlap_ratio));

    sem_times = [];
    sem_indices = [];

    for start_idx = 1:step_len:(L - window_len + 1)
        idx_range = start_idx:(start_idx + window_len - 1);
        loc_seg = loc(idx_range);
        roc_seg = roc(idx_range);

        sum_eog = loc_seg + roc_seg;
        hannWin = hann(window_len)';
        loc_win = loc_seg .* hannWin;
        roc_win = roc_seg .* hannWin;
        diff_eog = loc_win - roc_win;

        % Filtering
        b1_loc = filter(band1Filt, loc_win);
        b1_roc = filter(band1Filt, roc_win);
        b2_loc = filter(band2Filt, loc_win);
        b2_roc = filter(band2Filt, roc_win);
        b3_diff = filter(band3Filt, diff_eog);

        % Square + moving average
        squared_diff = b3_diff.^2;
        smooth_kernel = ones(1, round(0.1 * fs)) / round(0.1 * fs);
        smoothed_diff = conv(squared_diff, smooth_kernel, 'same');

        % Feature extraction
        peak_diff = max(smoothed_diff);
        peak_sum = max(abs(sum_eog));

        if all(std([b1_loc; b1_roc], 0, 2) > 1e-3)
            corr1 = corr(b1_loc', b1_roc');
        else
            corr1 = 0;
        end
        if all(std([b2_loc; b2_roc], 0, 2) > 1e-3)
            corr2 = corr(b2_loc', b2_roc');
        else
            corr2 = 0;
        end

        SEM_feature = corr2 - corr1;

        if (SEM_feature > 0.1) && (corr1 < 0) && (peak_sum > 50)
            t_center = (start_idx + window_len / 2) / fs;
            sem_times(end+1) = t_center;
            sem_indices{end+1} = idx_range;
        end
    end

    %% SEM pattern analysis
    if length(sem_times) >= 2
        ISI = diff(sem_times);

        % Burst definition: ISI < 3 seconds → 同一组
        burst_starts = [1, find(ISI >= 3) + 1];
        burst_ends = [burst_starts(2:end)-1, length(sem_times)];
        burst_sizes = burst_ends - burst_starts + 1;
    else
        ISI = [];
        burst_sizes = [];
    end

    sem_pattern.ISI = ISI;
    sem_pattern.density = length(sem_times);  % 留给主程序除以分钟
    sem_pattern.burst_count = sum(burst_sizes > 1);
    sem_pattern.burst_sizes = burst_sizes;
end
