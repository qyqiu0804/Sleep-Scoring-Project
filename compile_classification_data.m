function [X,Y] = compile_classification_data(datadir, edfFiles, xmlFiles,eeg_band_freqs,range)
X=[]; Y=[];
for k = range

    %% start with reading the basics from the files
    [hdr, record] = edfread([datadir filesep edfFiles(k).name]); % Read EDF file
    [events, stages, epochLength, annotation] = readXML([datadir filesep xmlFiles(k).name]);

    %% EEG processing
    EEG_channel = find(ismember(hdr.label,'EEG')); % find EEG channel -- should be 8
    fs_eeg = hdr.samples(EEG_channel);                 % hdr.samples is an array of sample rates for each channel
    EEG_signal = record(EEG_channel, :);           % extract EEG signal
    % apply preprocessing to data (like filtering)
    EEG_filtered_signal = preprocess_EEG(EEG_signal, fs_eeg);
    % this function computes and bundles the features for EEG
    eeg_features = extract_features_eeg(EEG_filtered_signal, fs_eeg, eeg_band_freqs);

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
    num_observations = size(eeg_features,2); % get number of training examples out of eeg processing
    eog_features = extract_features_eog(EOG_filtered_signal, fs_eog, num_observations); % last eog obs are cut off to match eeg rows
    
    %% Combine features that we will use in the classifier
    combined_features = [eeg_features;eog_features]; % Stack features
    % Align with sleep stages
    stages_at_epoch = stages(1:epochLength:end);
    num_epochs = length(stages_at_epoch);
    len = min(num_epochs, size(combined_features,2));
    % Store data
    X = [X; combined_features(:,1:len)'];  % Transpose for correct shape
    Y = [Y; stages_at_epoch(1:len)'];      % So that now rows are observations, columns are different features
end
