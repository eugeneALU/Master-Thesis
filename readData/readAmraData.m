function results = readAmraData(file)
% Reads data from Patrik at Amra. Data should be in a .csv-file.
% 
% Input: Name/path to the .csv-file.
% Output: A cellarray containing all the data.
%
% Created 2016-05-13 by Markus Karlsson

fid = fopen(file,'r');
header = fgetl(fid);
results(1,:) = strsplit(header,';');

while ~feof(fid)
    dataLine = fgetl(fid);
    dataLineSplit = strread(dataLine,'%s','delimiter',';')';
    temp = cell(1,size(results,2));
    
    for i = 1:length(dataLineSplit)
        if isempty(str2num(dataLineSplit{i}))
            temp{i} = dataLineSplit{i};
        else
            temp{i} = str2num(dataLineSplit{i});
        end
    end
    
    results = [results; temp];
end

fclose(fid);