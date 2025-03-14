function [ar_coeffs, PSD, F, best_order] = compute_ar_psd(signal, fs, max_order, Nfft)
    % Compute optimal AR model order using AIC
    N = length(signal);
    aic_values = zeros(1, max_order);
    
    for order = 1:max_order
        [~, E] = aryule(signal, order);
        if isnan(E) || E <= 0
            continue;
        end
        aic_values(order) = N * log(E) + 2 * order;
    end
    
    % Select the best AR model order using AIC
    [~, best_order] = min(aic_values);
    best_order = min(max(best_order, 4), 15);

    % Estimate AR coefficients using best order
    ar_coeffs = aryule(signal, best_order);

    % Compute PSD using AR coefficients
    [PSD, F] = pyulear(ar_coeffs, best_order, Nfft, fs);

    % Optional: Plot AIC values for reference
    figure;
    plot(1:max_order, aic_values, 'o-', 'LineWidth', 1.5);
    xlabel('AR Model Order');
    ylabel('AIC Value');
    title('AIC for Different AR Model Orders');
    grid on;
end
