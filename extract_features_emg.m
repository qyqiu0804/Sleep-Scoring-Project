function y = extract_features_emg(emg, fs, num_steps)
stepSeconds = 30; % 30 seconds to match the truth "epoch" rate
stepSamples = round(stepSeconds*fs);
totalSamples = stepSamples*num_steps;
emgMatrix = reshape(emg(1:totalSamples),stepSamples,num_steps);

emgRms = sqrt(sum(emgMatrix.^2)); % root mean square

emgRmsDiff = sqrt(sum(diff(emgMatrix).^2)); % root mean square of derivative

[spec_snr, entropy, binWidth, freqs] = extract_features_spectrogram(emg,fs,30,71,num_steps); % short time Fourier analysis
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

thresh = sum(spec_snr > 10); %arbitrary number, picked 10 (was 6) since data is rms filtered so pretty smooth


burst_threshold = 5*0.5492;  %std(diff(EMG_signal))=0.5492 so went 5x that
burst_detection = zeros(1, num_steps);
for i = 1:num_steps
    epoch = emgMatrix(:, i);
    burst_detection(i) = sum(abs(diff(epoch)) > burst_threshold);  % Count bursts in epoch
end

% Muscle tone analysis (mean and std of the filtered signal)
muscle_tone_mean = mean(emgMatrix);  % Average muscle tone in epoch
muscle_tone_std = std(emgMatrix);    % Variability in muscle tone in epoch

y = [emgRms;emgRmsDiff;entropy;thresh;band_entropy;burst_detection;muscle_tone_mean;muscle_tone_std];
