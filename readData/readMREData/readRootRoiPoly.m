function [poly] = readRootRoiPoly(fileName, resolution, roiNumberToUse)
% Reads a root-ROI and returns a mask
%
% [poly] = readRootRoi(fileName, resolution, roiNumberToUse)
%
% Created by Markus Karlsson, 2016-03-18

file1 = fopen(fileName, 'r');
if file1 == -1
    error(['Could not open file: ' fileName])
end
storeRoi = cell(1,1);

while ~feof(file1)
    line1 = fgetl(file1);
    
    % 0 0 at the end of the file
    if strcmp(line1,'0 0')
        break
    elseif length(line1) == 3 % This line should have 3 characters: 'X Y' (X=slice, Y=roi)
        sliceNumber = str2num(line1(1));
        roiNumber = str2num(line1(3));
        % Reading how many coordinates to come
        line2 = fgetl(file1);
        no_pairs = str2num(line2);
        
        if roiNumber == roiNumberToUse   % Only interested in looking at one roi
            
            roi = zeros(no_pairs,2);
            for i = 1:no_pairs
                line3 = fgetl(file1);
                roi(i,:) = str2num(line3);
            end
            storeRoi{sliceNumber} = roi;
            
        else % Wrong roi just read it all
            for i = 1:no_pairs
                line3 = fgetl(file1);
            end
        end
    else
        error(['Read a line that is just wrong: "' line1 '"'])
    end
    
end

poly = root_to_pixel(storeRoi{5},resolution);

end

function [pixel_roi] = root_to_pixel(root_roi, resolution)
% Assuming that the scale in root is "smooth" from 0 to 3.2

dim_x = resolution(1);
dim_y = resolution(2);
root_scale_x = 3.2;
root_scale_y = 3.2;
factor_x = dim_x/root_scale_x;
factor_y = dim_y/root_scale_y;

pixel_roi(:,1) = round(root_roi(:,1)*factor_x);
pixel_roi(:,2) = round(root_roi(:,2)*factor_y);

end