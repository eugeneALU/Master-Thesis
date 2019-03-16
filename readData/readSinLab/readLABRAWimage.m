function [ labraw ] = readLABRAWimage( filename )
%readLABRAWimage Reads philips LABRAW file to matlabstruct
%   To be developed

% Allow user to select a file if input FILENAME is not provided or is empty
if isempty(filename)
    [fn, pn] = uigetfile({'*.raw'},'Select a RAW file');
    if fn ~= 0
        filename = sprintf('%s%s',pn,fn);
    else
        disp('labraw2mat cancelled');
        return;
    end
end

[data,info] = loadLABRAW(filename);


labraw.data = data;
labraw.info = info;
