function outImage = readMREOutput(fileName,resolution,varargin)
% Reads one or more .map slices output files from MRE.
%
% outImage = readMREOutput(fileName,resolution)
%
% Dedault will read 9 slices
%
% fileName: String with the file name of the .map file, excluding the last
% underscore and slice nmber
% resolution: the resolution of the .map file
% outImage: The read image
%
% outImage = readMREOutput(fileName,resolution,slices)
%
% slices: A vector with the slices that will be read. Slices must be
% concexutive.
%
% Examples:
% GdImage = readMREOutput('exIm_16Hz',[80 80]);
% GdImage = readMREOutput('exIm_16Hz',[80 80],5);
% GdImage = readMREOutput('exIm_16Hz',[80 80],1:9);
%
% Created by Markus Karlsson, 2016-03-18

if nargin == 2
    slices = 1:9;
else
    slices = varargin{1};
end

nSlices = length(slices);
tempImage = zeros([resolution nSlices]);

for i = 1:nSlices
    sliceToOpen = num2str(slices(i));
    openName = [fileName '_' sliceToOpen '.map'];
    file = fopen(openName,'r');
    if file == -1
        error(['Could not open file: ' openName])
    end
    formatSpec = '%f';
    % Reads all the values in Pa
    vector = fscanf(file,formatSpec);
    fclose(file);
    tempImage(:,:,i) = reshape(vector,resolution);
end

outImage = tempImage;

end