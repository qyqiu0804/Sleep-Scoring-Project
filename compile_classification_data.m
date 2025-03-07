function [X,Y] = compile_classification_data(datadir, edfFiles, xmlFiles,band_freqs,range)
X=[]; Y=[];
for k = range

    %start with reading the basics from the files
    [hdr, record] = edfread([datadir filesep edfFiles(k).name]); % Read EDF file
    [events, stages, epochLength, annotation] = readXML([datadir filesep xmlFiles(k).name]);
    EEG_channel = find(ismember(hdr.label,'EEG')); % find EEG channel -- should be 8
    fs_EEG = hdr.samples(EEG_channel);                 % hdr.samples is an array of sample rates for each channel
    EEG_signal = record(EEG_channel, :);           % extract EEG signal

    %apply preprocessing to data
    EEG_filtered_signal = preprocess_EEG(EEG_signal, fs_EEG);

    %now extracting features

    %Starting with EEG features
    %first we use the short time fourier to get input for features we want
    %to use
    [spec_pwr, specBinWidthHz, freqs] = extract_features_spectrogram(EEG_filtered_signal,fs_EEG); % short time Fourier analysis
    spec_snr = spec_pwr ./ median(spec_pwr,2);     % normalization: equalize FFT bins  based on their background level
    %IMPORTANT SECTION - call functions for EEG features to be used in
    %classifier here
    band_ratios = compute_band_power_ratios(spec_snr, specBinWidthHz, band_freqs); % sofia task - compute relative power of each band
    %entropy = spectralEntropy(10*log10(max(spec_snr,1)),freqs).'; %sofia task - entropy
    band_entropies = compute_band_entropies(spec_snr, specBinWidthHz, band_freqs,freqs); %sofia task- band entropy
    %[~,peak_freq_inds] = max(spec_snr); peak_freqs = freqs(peak_freq_inds)'; %sofia task - peak frequency
    
    % Combine features that we will use in the classifier
    combined_features = [band_ratios;band_entropies]; % Stack features

   % Align with sleep stages
    stages_at_epoch = stages(1:epochLength:end);
    num_epochs = length(stages_at_epoch);
    len = min(num_epochs, size(combined_features,2));

    % Store data
    X = [X; combined_features(:,1:len)'];  % Transpose for correct shape
    Y = [Y; stages_at_epoch(1:len)'];
end
