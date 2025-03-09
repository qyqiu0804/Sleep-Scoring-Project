function y = extract_features_eog(eog, fs, num_steps)
stepSeconds = 30; % 30 seconds to match the truth "epoch" rate
stepSamples = round(stepSeconds*fs);
totalSamples = stepSamples*num_steps;
eogMatrix = reshape(eog(1:totalSamples),stepSamples,num_steps);

eogRms = sqrt(sum(eogMatrix.^2)); % root mean square

a = eogMatrix(1:stepSamples-1,:);
b = eogMatrix(2:stepSamples,:);
eogZeroCross = sum(sign(a) ~= sign(b)); % count zero crossings

eogRmsDiff = sqrt(sum(diff(eogMatrix).^2)); % root mean square of derivative

[spec_snr, entropy, binWidth, freqs] = extract_features_spectrogram(eog,fs,30,71,num_steps); % short time Fourier analysis
entropy = smooth(entropy,5)';

num_bands = 4;
band_entropy = zeros(num_bands,length(entropy));
spec_bins = size(spec_snr,1); % number of bins on one slice of a spectrum (FFT)
band_size = round(spec_bins/num_bands);
for k = 1:num_bands
    offset = (k-1)*band_size;
    range = (offset+1) : min(offset+band_size,spec_bins);
    band_entropy(k,:) = smooth(spectralEntropy(10*log10(max(spec_snr(range,:),1)),freqs(range)),6).';
end

thresh = sum(spec_snr > 6);

y = [eogRms;eogRmsDiff;eogZeroCross;entropy;thresh;band_entropy];
