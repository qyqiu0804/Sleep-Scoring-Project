function y = extract_features_eog(eogL, eogR, fs, num_steps)
    stepSeconds = 30;
    stepSamples = round(stepSeconds * fs);
    totalSamples = stepSamples * num_steps;
    eogAvg = (eogL + eogR) / 2;
    eogMatrix = reshape(eogAvg(1:totalSamples), stepSamples, num_steps);

    %% Time-Domain and Frequency-Domain Features (from averaged signal)
    eogRms = sqrt(sum(eogMatrix.^2));
    eogRmsDiff = sqrt(sum(diff(eogMatrix).^2));
    eogZeroCross = sum(sign(eogMatrix(1:end-1, :)) ~= sign(eogMatrix(2:end, :)));

    [spec_snr, entropy, ~, freqs] = extract_features_spectrogram(eogAvg, fs, 30, 71, num_steps);
    entropy = smooth(entropy, 5)';

    num_bands = 4;
    band_entropy = zeros(num_bands, length(entropy));
    spec_bins = size(spec_snr, 1);
    band_size = round(spec_bins / num_bands);
    for k = 1:num_bands
        offset = (k - 1) * band_size;
        range = (offset+1) : min(offset + band_size, spec_bins);
        band_entropy(k, :) = smooth(spectralEntropy(10*log10(max(spec_snr(range,:), 1)), freqs(range)), 6).';
    end

    thresh = sum(spec_snr > 6);

    %% Blink Features (based on averaged EOG)
    blink_feat = zeros(7, 1);
    nz = find(abs(preprocess_EOG(eogAvg, fs)) > 10);
    if ~isempty(nz)
        s = min(nz); e = max(nz);
        valid = eogAvg(s:e);
        dur = length(valid) / fs;
        info = detect_blinks(preprocess_EOG(valid, fs), fs);
        blink_feat = [
            length(info.times) / (dur / 60);
            safe_mean(info.durations);
            safe_std(info.durations);
            safe_mean(info.amplitudes);
            safe_std(info.amplitudes);
            safe_mean(info.IBI);
            safe_std(info.IBI)
        ];
    end

    %% SEM and REM (based on both eyes)
    semrem_feat = zeros(10, 1);
    nzL = find(abs(preprocess_EOG(eogL, fs)) > 10);
    nzR = find(abs(preprocess_EOG(eogR, fs)) > 10);
    if ~isempty(nzL) && ~isempty(nzR)
        s = max(min(nzL), min(nzR));
        e = min(max(nzL), max(nzR));
        loc = preprocess_EOG(eogL(s:e), fs);
        roc = preprocess_EOG(eogR(s:e), fs);
        dur = length(loc) / fs;

        [sem_times, ~, sem_pattern] = detect_sem_sliding(loc, roc, fs);
        [rem_times, ~, rem_pattern] = detect_rem_sliding(loc, roc, fs);

        sem_density = length(sem_times) / (dur / 60);
        rem_density = length(rem_times) / (dur / 60);

        semrem_feat = [
            sem_density;
            safe_mean(sem_pattern.ISI);
            safe_std(sem_pattern.ISI);
            safe_value(sem_pattern.burst_count);
            safe_mean(sem_pattern.burst_sizes);
            rem_density;
            safe_mean(rem_pattern.IRI);
            safe_std(rem_pattern.IRI);
            safe_value(rem_pattern.burst_count);
            safe_mean(rem_pattern.burst_sizes)
        ];
    end

    %% Final Output (repeat static features to match num_steps)
    features_static = [blink_feat; semrem_feat];
    features_static = repmat(features_static, 1, num_steps);

    y = [eogRms; eogRmsDiff; eogZeroCross; entropy; thresh; band_entropy; features_static];
end

%% Utility functions
function m = safe_mean(x)
    if isempty(x); m = 0; else; m = mean(x); end
end

function s = safe_std(x)
    if isempty(x); s = 0; else; s = std(x); end
end

function v = safe_value(x)
    if isempty(x); v = 0; else; v = x; end
end
