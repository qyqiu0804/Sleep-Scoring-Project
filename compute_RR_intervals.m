function RR_intervals = compute_RR_intervals(R_peaks, fs)
% compute_RR_intervals - Calculate RR intervals from R peaks
% Inputs: R_peaks - indices of R peaks, fs - sampling frequency
% Outputs: RR_intervals - vector of RR intervals in seconds

RR_intervals = diff(R_peaks) / fs;
end
