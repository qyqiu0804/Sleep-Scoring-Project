function band_ratios = compute_band_power_ratios(spec_pwr, specBinWidthHz, band_freqs)
    % Compute power in each band
    band_powers = sum_band_powers(spec_pwr, specBinWidthHz, band_freqs);

    % Compute total power (sum across all bands)
    total_power = sum(band_powers, 1);

    % Compute relative power ratios
    band_ratios = band_powers ./ total_power;
end