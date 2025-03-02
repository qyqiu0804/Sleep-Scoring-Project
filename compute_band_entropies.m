function band_entropies = compute_band_entropies(spec_pwr, specBinWidthHz, band_freqs,freqs)

num_bands = size(band_freqs,1);
[num_freqs,time_steps] = size(spec_pwr);

band_entropies = zeros(num_bands, time_steps);

for k = 1:num_bands
    band_inds = round(band_freqs(k,:)/specBinWidthHz) + 1;
    range = band_inds(1):band_inds(2);
    band_entropies(k,:) = spectralEntropy(10*log10(max(spec_pwr(range,:),1)),freqs(range)).';
end
