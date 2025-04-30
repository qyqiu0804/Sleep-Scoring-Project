function [X,Y] = compile_classification_data(datadir, edfFiles, xmlFiles, eeg_band_freqs, range)
X = []; Y = [];

for k = range
    %% Read EDF and XML
    [hdr, record] = edfread([datadir filesep edfFiles(k).name]);
    [events, stages, epochLength, annotation] = readXML([datadir filesep xmlFiles(k).name]);

    %% EEG Processing
    EEG_channel = find(ismember(hdr.label, 'EEG'));
    fs_eeg = hdr.samples(EEG_channel);
    EEG_signal = record(EEG_channel, :);
    EEG_filtered_signal = preprocess_EEG(EEG_signal, fs_eeg);
    eeg_features = extract_features_eeg(EEG_filtered_signal, fs_eeg, eeg_band_freqs);

    %% EOG Processing (Left + Right separately)
    EOGL_channel = find(ismember(hdr.label, 'EOGL'));
    EOGR_channel = find(ismember(hdr.label, 'EOGR'));
    EOGL_signal = record(EOGL_channel, :);
    EOGR_signal = record(EOGR_channel, :);
    fs_eog = hdr.samples(EOGL_channel);  % assumed same for both eyes

    EOGL_filtered = preprocess_EOG(EOGL_signal, fs_eog);
    EOGR_filtered = preprocess_EOG(EOGR_signal, fs_eog);

    num_observations = size(eeg_features, 2);
    eog_features = extract_features_eog(EOGL_filtered, EOGR_filtered, fs_eog, num_observations);

    %% EMG Processing
    EMG_channel = find(ismember(hdr.label,'EMG')); 
    EMG_signal = record(EMG_channel, :);
    % Create time vector for EMG signal (in seconds)
    fs_emg = hdr.samples(EMG_channel) ;
    EMG_filtered_signal = preprocess_EMG(EMG_signal, fs_emg);
    emg_features = extract_features_emg(EMG_filtered_signal, fs_emg, num_observations); % last emg obs are cut off to match eeg rows
    
     %% ECG Processing
    ECG_channel = find(ismember(hdr.label,'ECG')); 
    ECG_signal = record(ECG_channel, :);
    % Create time vector for ECG signal (in seconds)
    fs_ecg = hdr.samples(ECG_channel) ;
    ECG_filtered_signal = preprocess_ECG(ECG_signal, fs_ecg);
    ecg_features = extract_features_ecg(ECG_filtered_signal, fs_emg, num_observations); % last emg obs are cut off to match eeg rows
    %% Combine features

    combined_features = [eeg_features; eog_features; emg_features;ecg_features];

    %% -Align with sleep stages
    stages_at_epoch = stages(1:epochLength:end);
    num_epochs = length(stages_at_epoch);
    len = min(num_epochs, size(combined_features,2));

    %% -ave to master matrix
    X = [X; combined_features(:,1:len)'];
    Y = [Y; stages_at_epoch(1:len)'];
end
