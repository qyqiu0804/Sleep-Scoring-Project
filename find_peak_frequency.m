function peak_freq = find_peak_frequency(F, PSD, band)
    % Select frequency range within specified band
    valid_F = F(F >= band(1) & F <= band(2));
    valid_PSD = PSD(F >= band(1) & F <= band(2));
    
    % Find peak frequency
    [~, idx] = max(valid_PSD);
    peak_freq = valid_F(idx);
end
