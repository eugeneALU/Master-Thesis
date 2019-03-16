function [ labraw ] = readLABRAWmrs( filename )
%labraw2mat Reads philips LABRAW file to matlabstruct
%   Developed 140731 Anders Tisell

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


labraw.Samples = info.dims.nKx;

dataIDX = 0;
for KyIDX = 1:info.dims.nKy, 
    for KzIDX = 1:info.dims.nKz, 
        for rowIDX = 1:info.dims.nRows
            
            dataIDX = dataIDX + 1;
             
            DataAverage = mean(mean(data,1),12);
            labraw.MetaboliteFID{dataIDX} = squeeze(DataAverage(1,:,KyIDX,KzIDX,1,1,1,1,1,rowIDX,1,1));
            if info.dims.nMixes == 2
                labraw.ReferenceFID{dataIDX} = squeeze(DataAverage(1,:,KyIDX,KzIDX,1,1,1,1,1,rowIDX,2,1));
                
            end
        end 
    end 
end

end

