function ecg_filtered = preprocess_ECG(ecg_signal, fs, method)
% preprocess_ECG - Preprocess ECG signal
% Inputs:
%   ecg_signal - Raw ECG signal
%   fs - Sampling frequency (Hz)
%   method - 'butter' (default) for traditional filtering, or 'wavelet' for wavelet-based baseline removal
% Outputs:
%   ecg_filtered - Preprocessed ECG signal

if nargin < 3
    method = 'butter'; % Default method
end

switch lower(method)
    case 'butter'
        %% Method 1: Traditional Filtering
        % 1. High-pass filter to remove baseline wander
        hp_cutoff = 0.5; % High-pass cutoff frequency in Hz
        [b_hp, a_hp] = butter(2, hp_cutoff / (fs / 2), 'high');
        ecg_hp = filtfilt(b_hp, a_hp, ecg_signal);

        % 2. Low-pass filter to remove high-frequency noise
        lp_cutoff = 40; % Low-pass cutoff frequency in Hz
        [b_lp, a_lp] = butter(4, lp_cutoff / (fs / 2), 'low');
        ecg_filtered = filtfilt(b_lp, a_lp, ecg_hp);

    case 'wavelet'
        %% Method 2: Wavelet-based Baseline Removal
        level = 6;          % Decomposition level
        wavelet = 'db6';    % Daubechies 6 wavelet, good for ECG
        [C,L] = wavedec(ecg_signal, level, wavelet);

        % Set approximation coefficients (low frequency) to zero
        approx = appcoef(C, L, wavelet, level);
        C(1:length(approx)) = 0;

        % Reconstruct the signal
        ecg_filtered = waverec(C, L, wavelet);

        % Optional: low-pass filtering after wavelet to further denoise
        lp_cutoff = 40; % Hz
        [b_lp, a_lp] = butter(4, lp_cutoff / (fs / 2), 'low');
        ecg_filtered = filtfilt(b_lp, a_lp, ecg_filtered);

    otherwise
        error('Unknown method. Use ''butter'' or ''wavelet''.');
end
