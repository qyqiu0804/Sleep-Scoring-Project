function [rem_times, rem_indices, rem_pattern] = detect_rem_sliding(loc, roc, fs, window_sec, overlap_ratio)
% Strict REM detection based on SEM-like sliding window framework

    arguments
        loc (1,:) double
        roc (1,:) double
        fs (1,1) double
        window_sec (1,1) double = 1.5
        overlap_ratio (1,1) double = 0.75
    end

    % Design bandpass filters
    band1Filt = designfilt('bandpassiir','FilterOrder',4, ...
        'HalfPowerFrequency1',0.5,'HalfPowerFrequency2',6, ...
        'SampleRate',fs);
    band2Filt = designfilt('bandpassiir','FilterOrder',4, ...
        'HalfPowerFrequency1',1,'HalfPowerFrequency2',6, ...
        'SampleRate',fs);

    % Setup
    L = length(loc);
    window_len = round(window_sec * fs);
    step_len = round(window_len * (1 - overlap_ratio));

    rem_times = [];
    rem_indices = {};

    for start_idx = 1:step_len:(L - window_len + 1)
        idx_range = start_idx:(start_idx + window_len - 1);
        loc_seg = loc(idx_range);
        roc_seg = roc(idx_range);

        sum_eog = loc_seg + roc_seg;
        hannWin = hann(window_len)';
        loc_win = loc_seg .* hannWin;
        roc_win = roc_seg .* hannWin;
        diff_eog = loc_win - roc_win;

        % Bandpass filtering
        b1_loc = filter(band1Filt, loc_win);
        b1_roc = filter(band1Filt, roc_win);
        b2_loc = filter(band2Filt, loc_win);
        b2_roc = filter(band2Filt, roc_win);

        % Correlations
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

        REM_feature = corr1 - corr2;
        peak_sum = max(abs(sum_eog));

        % Detection rule for REM
        if (REM_feature > 0.09) && (corr1 > 0.7) && (peak_sum > 40)
            t_center = (start_idx + window_len / 2) / fs;
            rem_times(end+1) = t_center;
            rem_indices{end+1} = idx_range;
        end
    end

    % ---- REM Pattern Analysis ----
    rem_pattern = struct();
    if length(rem_times) > 1
        rem_pattern.IRI = diff(rem_times);  % Inter-REM Interval
        burst_thresh = 2;  % seconds
        burst_count = 1;
        burst_sizes = 1;

        for i = 2:length(rem_pattern.IRI)
            if rem_pattern.IRI(i) <= burst_thresh
                burst_sizes(end) = burst_sizes(end) + 1;
            else
                burst_count = burst_count + 1;
                burst_sizes(end+1) = 1;
            end
        end

        rem_pattern.burst_count = burst_count;
        rem_pattern.burst_sizes = burst_sizes;
    else
        rem_pattern.IRI = [];
        rem_pattern.burst_count = 0;
        rem_pattern.burst_sizes = [];
    end
end
