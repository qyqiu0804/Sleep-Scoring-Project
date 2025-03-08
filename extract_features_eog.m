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

specWinSeconds = 71;  % duration of spectrogram's moving window in seconds
specStepSeconds = 30; % how much to step the window for each output column
specWinSamples = round(specWinSeconds*fs);
specOverlapSamples = specWinSamples - round(specStepSeconds*fs);
specWin = hanning(specWinSamples); % window to suppress spectral sidelobes
[spec,freqs,times] = spectrogram(eog,specWin,specOverlapSamples,specWinSamples,fs);
spec = spec(:,1:num_steps);
spec_pwr = abs(spec).^2;              % notional power
spec_snr = spec_pwr./median(spec_pwr,2);

entropy = spectralEntropy(10*log10(max(spec_snr,1)),freqs).';
entropy = smooth(entropy,5)';

num_bands = 4;
band_entropy = zeros(num_bands,length(entropy));
band_size = round(specWinSamples/2/num_bands);
for k = 1:num_bands
    offset = (k-1)*band_size;
    range = (offset+1) : min(offset+band_size,specWinSamples/2+1);
    band_entropy(k,:) = smooth(spectralEntropy(10*log10(max(spec_snr(range,:),1)),freqs(range)),6).';
end

thresh = sum(spec_snr > 6);

y = [eogRms;eogRmsDiff;eogZeroCross;entropy;thresh;band_entropy];