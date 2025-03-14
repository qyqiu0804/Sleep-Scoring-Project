function ratios = compute_band_ratios(PSD_delta, PSD_theta, PSD_alpha, PSD_beta, F_delta, F_theta, F_alpha, F_beta)
    % Compute power in each band using AR model spectrum
    P_delta = trapz(F_delta, PSD_delta);
    P_theta = trapz(F_theta, PSD_theta);
    P_alpha = trapz(F_alpha, PSD_alpha);
    P_beta = trapz(F_beta, PSD_beta);
    
    % Compute total power
    P_total = P_delta + P_theta + P_alpha + P_beta;
    
    % Compute spectral band ratios
    ratios.delta = P_delta / P_total;
    ratios.theta = P_theta / P_total;
    ratios.alpha = P_alpha / P_total;
    ratios.beta = P_beta / P_total;
end
