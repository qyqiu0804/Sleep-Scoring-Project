function ar_features = extract_ar_features(EEG_signal, fs)
    
    % Define filter parameters
    filter_order = 4;
    delta_band = [0.5 4];
    theta_band = [4 8];
    alpha_band = [8 13];
    beta_band = [13 30];

    % Filter signals for each EEG band
    [delta_b, delta_a] = butter(filter_order, delta_band / (fs / 2), 'bandpass');
    delta_signal = filtfilt(delta_b, delta_a, EEG_signal);

    [theta_b, theta_a] = butter(filter_order, theta_band / (fs / 2), 'bandpass');
    theta_signal = filtfilt(theta_b, theta_a, EEG_signal);

    [alpha_b, alpha_a] = butter(filter_order, alpha_band / (fs / 2), 'bandpass');
    alpha_signal = filtfilt(alpha_b, alpha_a, EEG_signal);

    [beta_b, beta_a] = butter(filter_order, beta_band / (fs / 2), 'bandpass');
    beta_signal = filtfilt(beta_b, beta_a, EEG_signal);

    % Automatically select AR model order using AIC
    max_order = 30;
    aic_values = zeros(1, max_order);
    N = length(EEG_signal);

    for order_test = 1:max_order
        [~, E] = aryule(EEG_signal, order_test);
        if isnan(E) || E <= 0
            continue;
        end
        aic_values(order_test) = N * log(E) + 2 * order_test;
    end

    [~, best_order] = min(aic_values);
    best_order = min(max(best_order, 4), 15);

    % Compute AR model coefficients
    ar_coeffs.delta = aryule(delta_signal, best_order);
    ar_coeffs.theta = aryule(theta_signal, best_order);
    ar_coeffs.alpha = aryule(alpha_signal, best_order);
    ar_coeffs.beta = aryule(beta_signal, best_order);

    % Compute Power Spectral Density (PSD) using AR model
    Nfft = 1024;
    [PSD_delta, F_delta] = pyulear(ar_coeffs.delta, best_order, Nfft, fs);
    [PSD_theta, F_theta] = pyulear(ar_coeffs.theta, best_order, Nfft, fs);
    [PSD_alpha, F_alpha] = pyulear(ar_coeffs.alpha, best_order, Nfft, fs);
    [PSD_beta, F_beta] = pyulear(ar_coeffs.beta, best_order, Nfft, fs);

    % Compute power in each band
    P_delta = trapz(F_delta, PSD_delta);
    P_theta = trapz(F_theta, PSD_theta);
    P_alpha = trapz(F_alpha, PSD_alpha);
    P_beta = trapz(F_beta, PSD_beta);
    P_total = P_delta + P_theta + P_alpha + P_beta;

    % Compute spectral band ratios
    delta_ratio = P_delta / P_total;
    theta_ratio = P_theta / P_total;
    alpha_ratio = P_alpha / P_total;
    beta_ratio = P_beta / P_total;

    % Identify peak frequencies within each EEG band
    [~, delta_peak_idx] = max(PSD_delta(F_delta >= 0.5 & F_delta <= 4));
    delta_peak_freq = F_delta(delta_peak_idx);

    [~, theta_peak_idx] = max(PSD_theta(F_theta >= 4 & F_theta <= 8));
    theta_peak_freq = F_theta(theta_peak_idx);

    [~, alpha_peak_idx] = max(PSD_alpha(F_alpha >= 8 & F_alpha <= 13));
    alpha_peak_freq = F_alpha(alpha_peak_idx);

    [~, beta_peak_idx] = max(PSD_beta(F_beta >= 13 & F_beta <= 30));
    beta_peak_freq = F_beta(beta_peak_idx);

    % Output as a single vector (for easy merging)
    ar_features = [delta_ratio; theta_ratio; alpha_ratio; beta_ratio; ...
                   delta_peak_freq; theta_peak_freq; alpha_peak_freq; beta_peak_freq];
end