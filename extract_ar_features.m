function ar_features = extract_ar_features(EEG_signal, fs)
    
    % ===== Define filter parameters =====
    filter_order = 4;
    delta_band = [0.5 4];
    theta_band = [4 8];
    alpha_band = [8 13];
    beta_band = [13 30];

    % ===== Filter signals for each EEG band =====
    [delta_b, delta_a] = butter(filter_order, delta_band / (fs / 2), 'bandpass');
    delta_signal = filtfilt(delta_b, delta_a, EEG_signal);

    [theta_b, theta_a] = butter(filter_order, theta_band / (fs / 2), 'bandpass');
    theta_signal = filtfilt(theta_b, theta_a, EEG_signal);

    [alpha_b, alpha_a] = butter(filter_order, alpha_band / (fs / 2), 'bandpass');
    alpha_signal = filtfilt(alpha_b, alpha_a, EEG_signal);

    [beta_b, beta_a] = butter(filter_order, beta_band / (fs / 2), 'bandpass');
    beta_signal = filtfilt(beta_b, beta_a, EEG_signal);

    % ===== Fixed AR model order =====
    fixed_order = 6;

    % ===== Generate AR coefficients for each frequency band =====
    ar_coeffs.delta = aryule(delta_signal, fixed_order);
    ar_coeffs.theta = aryule(theta_signal, fixed_order);
    ar_coeffs.alpha = aryule(alpha_signal, fixed_order);
    ar_coeffs.beta = aryule(beta_signal, fixed_order);

    % ===== Compute PSD values using AR model =====
    Nfft = 1024;
    [PSD_delta, ~] = pyulear(ar_coeffs.delta, fixed_order, Nfft, fs);
    [PSD_theta, ~] = pyulear(ar_coeffs.theta, fixed_order, Nfft, fs);
    [PSD_alpha, ~] = pyulear(ar_coeffs.alpha, fixed_order, Nfft, fs);
    [PSD_beta, ~] = pyulear(ar_coeffs.beta, fixed_order, Nfft, fs);

    % ===== Take mean and max PSD values =====
    mean_PSD_delta = mean(PSD_delta);
    mean_PSD_theta = mean(PSD_theta);
    mean_PSD_alpha = mean(PSD_alpha);
    mean_PSD_beta = mean(PSD_beta);

    max_PSD_delta = max(PSD_delta);
    max_PSD_theta = max(PSD_theta);
    max_PSD_alpha = max(PSD_alpha);
    max_PSD_beta = max(PSD_beta);

% ===== Remove the first AR coefficient (always 1) =====
delta_ar = ar_coeffs.delta(2:end);
theta_ar = ar_coeffs.theta(2:end);
alpha_ar = ar_coeffs.alpha(2:end);
beta_ar = ar_coeffs.beta(2:end);


    % ===== Concatenate all AR coefficients and PSD values =====
    ar_features = [delta_ar, theta_ar, alpha_ar, beta_ar, ...
                   mean_PSD_delta, mean_PSD_theta, mean_PSD_alpha, mean_PSD_beta, ...
                   max_PSD_delta, max_PSD_theta, max_PSD_alpha, max_PSD_beta];

    % ===== Convert to row vector =====
    ar_features = ar_features(:)';

    % ===== Pad or truncate to consistent length =====
    fixed_len = 32;  % 4 bands × (6 AR coeffs) + 8 PSD features = 32
    if length(ar_features) < fixed_len
        ar_features = padarray(ar_features, [0, fixed_len - length(ar_features)], NaN, 'post');
    elseif length(ar_features) > fixed_len
        ar_features = ar_features(1:fixed_len);
    end
end
