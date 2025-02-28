% sum_band_powers
% inputs:
%   spec_pwr - the spectrogram power matrix [frequencies X time_steps]
%   specBinWidthHz - the spectrum (fft) bin width in Hertz
%   band_freqs - matrix of band edges [num_bands X 2]
function band_powers = sum_band_powers(spec_pwr, specBinWidthHz, band_freqs)
num_bands = size(band_freqs,1);
[num_freqs,time_steps] = size(spec_pwr);

band_powers = zeros(num_bands, time_steps);

for k = 1:num_bands
    band_inds = round(band_freqs(k,:)/specBinWidthHz) + 1;
    band_powers(k,:) = sum(spec_pwr(band_inds(1):band_inds(2),:));
end
