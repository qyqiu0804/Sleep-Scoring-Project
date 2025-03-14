function eeg_features = extract_features_eeg(EEG_filtered_signal, fs, band_freqs)

    %now extracting features
    %first we use the short time fourier to get input for features we want
    %to use
    [spec_snr, entropy, specBinWidthHz, freqs] = extract_features_spectrogram(EEG_filtered_signal,fs,30,99); % short time Fourier analysis
    %IMPORTANT SECTION
    %now these are the additional features we want to use in the classifier for eeg
    entropy = smooth(entropy,21)'; % smooth the total spectral entropy to reduce local fluctuations

    band_ratios = compute_band_power_ratios(spec_snr, specBinWidthHz, band_freqs); % sofia task - compute relative power of each band

    [~,peak_freq_inds] = max(spec_snr); peak_freqs = freqs(peak_freq_inds)';

    band_entropies = compute_band_entropies(spec_snr, specBinWidthHz, band_freqs,freqs);
    for k2 = 1:6, band_entropies(k2,:) = smooth(band_entropies(k2,:),20); end

    time_freq_features = extract_time_frequency_features(EEG_filtered_signal, fs, 99, 30);

    eeg_features = [band_ratios;entropy;peak_freqs;band_entropies; time_freq_features'];