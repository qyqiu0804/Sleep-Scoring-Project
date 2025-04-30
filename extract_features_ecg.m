function y = extract_features_ecg(ecg, fs, num_steps)
% EXTRACT_FEATURES_ECG
% Extract time-domain HRV features from ECG signal.
% Returns a [4 x num_steps] matrix:
% [SDNN; RMSSD; pNN50; mean_HR]

stepSeconds = 30; % Epoch length in seconds
stepSamples = round(stepSeconds * fs);
totalSamples = stepSamples * num_steps;

% Reshape the ECG into epochs [samples per epoch x num_steps]
ecgMatrix = reshape(ecg(1:totalSamples), stepSamples, num_steps);

% Initialize feature vectors
sdnn = zeros(1, num_steps);
rmssd = zeros(1, num_steps);
pnn50 = zeros(1, num_steps);
mean_hr = zeros(1, num_steps);

% Loop through each epoch
for i = 1:num_steps
    segment = ecgMatrix(:, i);

    % Detect R-peaks
    R_peaks = detect_R_peaks(segment, fs);

    % Compute RR intervals (in seconds)
    RR_intervals_s = compute_RR_intervals(R_peaks, fs);

    % Skip epoch if not enough RR intervals
    if length(RR_intervals_s) < 2
        sdnn(i) = NaN;
        rmssd(i) = NaN;
        pnn50(i) = NaN;
        mean_hr(i) = NaN;
        continue;
    end

    % Convert to milliseconds
    RR_intervals_ms = RR_intervals_s * 1000;

    % SDNN (standard deviation of NN intervals)
    sdnn(i) = std(RR_intervals_ms);

    % RMSSD (root mean square of successive differences)
    diff_RR = diff(RR_intervals_ms);
    rmssd(i) = sqrt(mean(diff_RR .^ 2));

    % pNN50 (percentage of successive RR intervals > 50 ms)
    pnn50(i) = 100 * sum(abs(diff_RR) > 50) / length(diff_RR);

    % Mean heart rate (in bpm), only using valid RR intervals (>0)
    valid_rr = RR_intervals_s(RR_intervals_s > 0);
    if isempty(valid_rr)
        mean_hr(i) = NaN;
    else
        mean_hr(i) = mean(60 ./ valid_rr);
    end
end

% Return feature matrix [4 x num_steps]
y = [sdnn; rmssd; pnn50; mean_hr];

end
