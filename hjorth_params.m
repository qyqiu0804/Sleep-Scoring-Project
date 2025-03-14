function [activity, mobility, complexity] = hjorth_params(signal)
    % Compute Activity (Variance)
    activity = var(signal);

    % Compute Mobility (Ratio of standard deviation of first derivative to the signal)
    diff_signal = diff(signal);
    mobility = std(diff_signal) / std(signal);

    % Compute Complexity (Ratio of mobility of first derivative to the signal's mobility)
    diff_diff_signal = diff(diff_signal);
    complexity = (std(diff_diff_signal) / std(diff_signal)) / mobility;
end