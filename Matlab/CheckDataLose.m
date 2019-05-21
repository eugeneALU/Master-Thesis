clear;
close all;
XlsxPath = ['..' filesep 'data xlsx' filesep];
ImagePath = ['..' filesep 'Image and Mask' filesep 'HIFI' filesep];

file = dir(ImagePath);
file= (file(4:end));
SIZE = max(size(file));


% for i = 1:SIZE
%     pid = file(i).name;
%     if (~isfile([XlsxPath pid '.xlsx']))
%         disp(['file ' pid '.xlsx not exist']);
%     end
% end

load('HIFIfibrosis.mat');
PID = {Fibrosis.PID}';

for i = 1:SIZE
    pid = file(i).name;
    if (~any(strcmp(PID, pid)))
        disp([pid ' do not have Fibrosis stage']);
    end
end
