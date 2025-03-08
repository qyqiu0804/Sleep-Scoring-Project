function [X,Y] = compile_classification_data(datadir, edfFiles, xmlFiles,band_freqs,range)
X=[]; Y=[];
for k = range

    %start with reading the basics from the files
    [hdr, record] = edfread([datadir filesep edfFiles(k).name]); % Read EDF file
    [events, stages, epochLength, annotation] = readXML([datadir filesep xmlFiles(k).name]);

    %% EEG processing
    EEG_channel = find(ismember(hdr.label,'EEG')); % find EEG channel -- should be 8
    fs = hdr.samples(EEG_channel);                 % hdr.samples is an array of sample rates for each channel
    EEG_signal = record(EEG_channel, :);           % extract EEG signal

    %apply preprocessing to data
    EEG_filtered_signal = preprocess_EEG(EEG_signal, fs);

    %now extracting features
    %first we use the short time fourier to get input for features we want
    %to use
    [spec_pwr, specBinWidthHz, freqs] = extract_features_spectrogram(EEG_filtered_signal,fs); % short time Fourier analysis
    spec_snr = spec_pwr ./ median(spec_pwr,2);     % normalization: equalize FFT bins based on their background level
    
    %IMPORTANT SECTION
    %now these are the features we want to use in the classifier
    band_ratios = compute_band_power_ratios(spec_snr, specBinWidthHz, band_freqs); % sofia task - compute relative power of each band

   
    entropy = spectralEntropy(10*log10(max(spec_snr,1)),freqs).';
    entropy = smooth(entropy,20)';

    [~,peak_freq_inds] = max(spec_snr); peak_freqs = freqs(peak_freq_inds)';

    band_entropies = compute_band_entropies(spec_snr, specBinWidthHz, band_freqs,freqs);
    for k2 = 1:6, band_entropies(k2,:) = smooth(band_entropies(k2,:),20); end

    %% EOG processing
    EOGL_channel = find(ismember(hdr.label,'EOGL')); 
    EOGL_signal = record(EOGL_channel, :);
    EOGR_channel = find(ismember(hdr.label,'EOGR')); 
    EOGR_signal = record(EOGR_channel, :);
    % Create time vector for EOG signal (in seconds)
    fs_eog = hdr.samples(EOGL_channel) ;% i checked that its the same for both eyes
    %Average the signals
    EOG_signal = (EOGL_signal + EOGR_signal) / 2;
    EOG_filtered_signal = preprocess_EOG(EOG_signal, fs_eog);
    num_observations = size(band_ratios,2); % number of rows is number of training examples
    eog_features = extract_features_eog(EOG_filtered_signal, fs_eog, num_observations); % last eog obs are cut off to match eeg rows
    
    % Combine features that we will use in the classifier
    combined_features = [band_ratios;entropy;peak_freqs;band_entropies;eog_features]; % Stack features

    % Align with sleep stages
    stages_at_epoch = stages(1:epochLength:end);
    num_epochs = length(stages_at_epoch);
    len = min(num_epochs, size(combined_features,2));

    % Store data
    X = [X; combined_features(:,1:len)'];  % Transpose for correct shape
    Y = [Y; stages_at_epoch(1:len)'];
end
