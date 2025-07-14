% 1. Citirea fisierului
filename = 'lung-cancer.data';
opts = detectImportOptions(filename, 'FileType', 'text', 'NumHeaderLines', 0);
opts.MissingRule = 'fill';
opts = setvartype(opts, 'char');  % Citim ca text

% 2. Încarcarea datelor
data = readtable(filename, opts);

% 3. Înlocuim '?' cu NaN si modificam in double
for i = 1:width(data)
    data.(i)(strcmp(data.(i), '?')) = {'NaN'};
    data.(i) = str2double(data.(i));
end

% 4. Elimina randurile cu valori lipsa
data = rmmissing(data);

% 5. Separarea datelor in caracteristici (X) si etichete (y)
X = table2array(data(:, 2:end));  % Coloanele 2 pânã la final => features
y = table2array(data(:, 1));      % Coloana 1 => label

% 6. Impartirea datelor în train/test (75/25)
cv = cvpartition(size(X,1), 'HoldOut', 0.25);
idx = cv.test;

XTrain = X(~idx, :);
YTrain = y(~idx);
XTest = X(idx, :);
YTest = y(idx);

% 7. Antrenarea modelului KNN (k = 5, distanta Euclidiana)
knnModel = fitcknn(XTrain, YTrain, 'NumNeighbors', 5, 'Distance', 'euclidean');

% 8. Predictii pe setul de test
YPred = predict(knnModel, XTest);

% 9. Calculul acuratetei
accuracy = sum(YPred == YTest) / numel(YTest);
fprintf("Test Accuracy: %.2f%%\n", accuracy * 100);

% 10. Matricea de confuzie
cm = confusionchart(YTest, YPred);
cm.Title = 'Confusion Matrix for KNN Lung Cancer Classification';
