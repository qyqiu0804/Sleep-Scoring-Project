function y = extract_features_EOG(EOG_signal, fs, num_steps)
stepSeconds = 30; % 30 seconds to match the truth "epoch" rate
stepSamples = round(stepSeconds*fs);
totalSamples = stepSamples*num_steps;
eogMatrix = reshape(EOG_signal(1:totalSamples),stepSamples,num_steps);

eogRms = sqrt(sum(eogMatrix.^2)); % root mean square
a = eogMatrix(1:stepSamples-1,:);
b = eogMatrix(2:stepSamples,:);
eogZeroCross = sum(sign(a) ~= sign(b)); % count zero crossings
eogRmsDiff = sqrt(sum(diff(eogMatrix).^2)); % root mean square of derivative

y = [eogRms;eogRmsDiff;eogZeroCross];
