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

    % ===== Automatically select AR model order using AIC =====
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
    best_order = min(max(best_order, 4), 15);  % Keep order between 4 and 15

    % ===== Generate AR coefficients for each frequency band =====
    ar_coeffs.delta = aryule(delta_signal, best_order);
    ar_coeffs.theta = aryule(theta_signal, best_order);
    ar_coeffs.alpha = aryule(alpha_signal, best_order);
    ar_coeffs.beta = aryule(beta_signal, best_order);

    % ===== Compute PSD values using AR model =====
    Nfft = 1024;
    [PSD_delta, F_delta] = pyulear(ar_coeffs.delta, best_order, Nfft, fs);
    [PSD_theta, F_theta] = pyulear(ar_coeffs.theta, best_order, Nfft, fs);
    [PSD_alpha, F_alpha] = pyulear(ar_coeffs.alpha, best_order, Nfft, fs);
    [PSD_beta, F_beta] = pyulear(ar_coeffs.beta, best_order, Nfft, fs);

    % ===== Take mean and max PSD values =====
    mean_PSD_delta = mean(PSD_delta);
    mean_PSD_theta = mean(PSD_theta);
    mean_PSD_alpha = mean(PSD_alpha);
    mean_PSD_beta = mean(PSD_beta);

    max_PSD_delta = max(PSD_delta);
    max_PSD_theta = max(PSD_theta);
    max_PSD_alpha = max(PSD_alpha);
    max_PSD_beta = max(PSD_beta);

    % ===== Pad AR coefficients to same length =====
    max_ar_len = max([length(ar_coeffs.delta), length(ar_coeffs.theta), length(ar_coeffs.alpha), length(ar_coeffs.beta)]);

    % Pad with NaN only if the length is shorter than max_ar_len
    if length(ar_coeffs.delta) < max_ar_len
        delta_ar = padarray(ar_coeffs.delta(:)', [0, max_ar_len - length(ar_coeffs.delta)], NaN, 'post');
    else
        delta_ar = ar_coeffs.delta(:)';
    end

    if length(ar_coeffs.theta) < max_ar_len
        theta_ar = padarray(ar_coeffs.theta(:)', [0, max_ar_len - length(ar_coeffs.theta)], NaN, 'post');
    else
        theta_ar = ar_coeffs.theta(:)';
    end

    if length(ar_coeffs.alpha) < max_ar_len
        alpha_ar = padarray(ar_coeffs.alpha(:)', [0, max_ar_len - length(ar_coeffs.alpha)], NaN, 'post');
    else
        alpha_ar = ar_coeffs.alpha(:)';
    end

    if length(ar_coeffs.beta) < max_ar_len
        beta_ar = padarray(ar_coeffs.beta(:)', [0, max_ar_len - length(ar_coeffs.beta)], NaN, 'post');
    else
        beta_ar = ar_coeffs.beta(:)';
    end

    % ===== Concatenate all AR coefficients and PSD values =====
    ar_features = [delta_ar, theta_ar, alpha_ar, beta_ar, ...
                   mean_PSD_delta, mean_PSD_theta, mean_PSD_alpha, mean_PSD_beta, ...
                   max_PSD_delta, max_PSD_theta, max_PSD_alpha, max_PSD_beta];

    % ===== Convert to row vector =====
    ar_features = ar_features(:)';

    % ===== Pad to consistent length (optional) =====
    % Ensure consistent feature vector length for classifier input
    max_len = 50;  % Adjust based on the model requirement

    if length(ar_features) < max_len
        % Only pad when the length is less than max_len
        ar_features = padarray(ar_features, [0, max_len - length(ar_features)], NaN, 'post');
    end

end
