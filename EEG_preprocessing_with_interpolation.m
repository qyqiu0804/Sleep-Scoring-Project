% Step 1: Load EDF and XML files
addpath('D:\Che Qiancao\KTH\Course\12\Signal\Lab\Project Data'); % Set the folder path containing the scripts

% Specify the path to your EDF and XML files
edfFilename = 'D:\Che Qiancao\KTH\Course\12\Signal\Lab\Project Data\R1.edf'; 
xmlFilename = 'D:\Che Qiancao\KTH\Course\12\Signal\Lab\Project Data\R1.xml';

% Read EDF file
[hdr, record] = edfread(edfFilename);

% Read XML file using readXML function
[events, stages, epochLength, annotation] = readXML(xmlFilename);
fs = 125; % Assuming the sampling frequency is the same across all channels

%% Process EEG Signal

% Extract the EEG signal from the 8th channel (assuming it's the EEG channel)
EEG_signal = record(8, :);

% Create time vector for EEG signal (in seconds)
time_vector = (0:length(EEG_signal)-1) / fs;

% Define the baseline segment (selecting the first 500 data points as the baseline segment)
baseline_segment = EEG_signal(1:500); 

% Calculate the mean of the baseline segment
baseline_mean = mean(baseline_segment);

% Remove baseline drift by subtracting the mean of the baseline segment from the entire signal
EEG_signal_baseline_corrected = EEG_signal - baseline_mean;

% Apply a Notch Filter (48-52 Hz) to remove power line interference
% for EU data
notch_filter_order = 2; % Filter order
notch_low = 48; % Lower cutoff frequency for the notch filter
notch_high = 52; % Upper cutoff frequency for the notch filter
[notch_b, notch_a] = butter(notch_filter_order, [notch_low, notch_high] / (0.5 * fs), 'stop');
EEG_signal_notch_filtered = filtfilt(notch_b, notch_a, EEG_signal_baseline_corrected);

% Apply a Bandpass Filter (0.3-35 Hz), range of eeg signals based on paper in canvas
bandpass_filter_order = 4; % Filter order
low_cutoff = 0.3; % Lower cutoff frequency for the bandpass filter
high_cutoff = 35; % Upper cutoff frequency for the bandpass filter
[bandpass_b, bandpass_a] = butter(bandpass_filter_order, [low_cutoff, high_cutoff] / (0.5 * fs), 'bandpass');
EEG_signal_bandpass_filtered = filtfilt(bandpass_b, bandpass_a, EEG_signal_notch_filtered);

% Extract SignalArtifactEvent from XML
eventConcepts = {events.EventConcept};
artifactIndices = strcmp(eventConcepts, 'SignalArtifactEvent');

% Extract Start and Duration of artifacts
artifactStartTimes = [events(artifactIndices).Start];
artifactDurations = [events(artifactIndices).Duration];

artifactStartSamples = round(artifactStartTimes * fs);
artifactEndSamples = round((artifactStartTimes + artifactDurations) * fs);

% Remove artifacts by setting affected samples to NaN
filtered_signal = EEG_signal_bandpass_filtered;
for i = 1:length(artifactStartSamples)
    filtered_signal(artifactStartSamples(i):artifactEndSamples(i)) = NaN;
end

% Interpolate missing values
nanIdx = isnan(filtered_signal);
x = 1:length(filtered_signal);
filtered_signal(nanIdx) = interp1(x(~nanIdx), filtered_signal(~nanIdx), x(nanIdx), 'linear');

% Create a figure with four subplots to compare all steps
figure;

% Plot the original EEG signal
subplot(4, 1, 1); 
plot(time_vector, EEG_signal);
title('Original EEG Signal');
xlabel('Time (s)');
ylabel('Amplitude');

% Plot the EEG signal after filtering
subplot(4, 1, 2); 
plot(time_vector, EEG_signal_bandpass_filtered);
title('EEG Signal after Notch and Bandpass Filtering (0.3-35 Hz)');
xlabel('Time (s)');
ylabel('Amplitude');

% Plot the EEG signal with artifacts removed
subplot(4, 1, 3); 
plot(time_vector, filtered_signal);
title('EEG Signal with Artifacts Removed');
xlabel('Time (s)');
ylabel('Amplitude');
