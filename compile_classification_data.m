function [X,Y] = compile_classification_data(datadir, edfFiles, xmlFiles,band_freqs,range)
X=[]; Y=[];
for k = range

    %start with reading the basics from the files
    [hdr, record] = edfread([datadir filesep edfFiles(k).name]); % Read EDF file
    [events, stages, epochLength, annotation] = readXML([datadir filesep xmlFiles(k).name]);

    % EEG processing
    EEG_channel = find(ismember(hdr.label,'EEG')); % find EEG channel
    fs_EEG = hdr.samples(EEG_channel);                 
    EEG_signal = record(EEG_channel, :);           % extract EEG signal
    EEG_filtered_signal = preprocess_EEG(EEG_signal, fs_EEG); %filter signal
    
    % EEG Features
    %first we use the short time fourier to get input for features we want
    %to use
    [spec_pwr, specBinWidthHz, freqs] = extract_features_spectrogram(EEG_filtered_signal,fs_EEG); % short time Fourier analysis
    spec_snr = spec_pwr ./ median(spec_pwr,2);     % normalization: equalize FFT bins based on their background level
    band_ratios = compute_band_power_ratios(spec_snr, specBinWidthHz, band_freqs); %feature- compute relative power of each band
    %entropy = spectralEntropy(10*log10(max(spec_snr,1)),freqs).'; %feature
    %[~,peak_freq_inds] = max(spec_snr); peak_freqs = freqs(peak_freq_inds)'; %feature- peak frequency
    band_entropies = compute_band_entropies(spec_snr, specBinWidthHz, band_freqs,freqs); %feature- entropy in each band

    % EOG processing
    EOGL_channel = find(ismember(hdr.label,'EOGL')); 
    EOGL_signal = record(EOGL_channel, :);
    EOGR_channel = find(ismember(hdr.label,'EOGR')); 
    EOGR_signal = record(EOGR_channel, :);
    fs_EOG = hdr.samples(EOGL_channel) ;%50
    EOG_signal = (EOGL_signal + EOGR_signal) / 2; %Average the signals
    EOG_filtered_signal = preprocess_EOG(EOG_signal, fs_EOG);
    num_observations = size(band_ratios,2); % number of rows is number of training examples
    
    % EOG_features
    EOG_features = extract_features_EOG(EOG_filtered_signal, fs_EOG, num_observations); % last eog obs are cut off to match eeg rows
    
    % Combine features that we will use in the classifier
    combined_features = [band_ratios;band_entropies;EOG_features]; % Stack features

    % Align with sleep stages
    stages_at_epoch = stages(1:epochLength:end);
    num_epochs = length(stages_at_epoch);
    len = min(num_epochs, size(combined_features,2));

    % Store data
    X = [X; combined_features(:,1:len)'];  % Transpose for correct shape
    Y = [Y; stages_at_epoch(1:len)'];
end
