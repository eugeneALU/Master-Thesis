function [ listdata] = listdata2mat( filename )
%labraw2mat Reads philips LABRAW file to matlabstruct
%
%   Developed 140731 Anders Tisell
%
%   listdata2mat calls [data,info] = readListDataAT(filename)
%   Where data is a matrix with k-space data, the number of dimentions is 
%   dependent on which of the follwoing attributs that are non zero  {'kx','ky','kz','loca','dyn','card','echo','mix','aver','chan','extr1','extr2'};



% Allow user to select a file if input FILENAME is not provided or is empty
if isempty(filename)
    [fn, pn] = uigetfile({'*.list'},'Select a LIST file');
    if fn ~= 0
        filename = sprintf('%s%s',pn,fn);
    else
        disp('listdata2mat2mat cancelled');
        return;
    end
end

[data,info] = readListDataAT(filename);

 
dataDim = size(data);
 

listdata.Samples = size(data,1);
if length(size(data)) >= 2, listdata.nKy = size(data,2); else listdata.nKy = 1; end,
if length(size(data)) >= 3, listdata.nkz = size(data,3); else listdata.nkz = 1; end,
if length(size(data)) >= 4, listdata.nloca = size(data,4); else listdata.nloca = 1; end,
if length(size(data)) >= 5, listdata.ndyn = size(data,5); else listdata.ndyn = 1; end,
if length(size(data)) >= 6, listdata.ncard = size(data,6); else listdata.ncard = 1; end,
if length(size(data)) >= 7, listdata.necho = size(data,7); else listdata.necho = 1; end,
if length(size(data)) >= 8, listdata.nmix = size(data,8); else listdata.nmix = 1; end,
if length(size(data)) >= 9, listdata.naver = size(data,9); data = mean(data,9); else listdata.naver = 1; end,
if length(size(data)) >= 10, listdata.nchan = size(data,10); data = mean(data,10); else listdata.nchan = 1; end,
if length(size(data)) >= 11, listdata.nextr1 = size(data,11); else listdata.nextr1 = 1; end,
if length(size(data)) >= 12, listdata.nextr2 = size(data,11); else listdata.nextr2 = 1; end,



dataIDX = 0;
for KyIDX = 1:listdata.nKy
    readIDX(2) = KyIDX;
    for KzIDX = 1:listdata.nKz 
        readIDX(3) = KzIDX;
        for 
        dataIDX = dataIDX + 1;
               if size(data,8)==2
                    listdata.ReferenceFID{dataIDX} = squeeze(DataAverage(:,KyIDX,KzIDX,1,1,1,1,1,rowIDX,2,1));
                    listdata.MetaboliteFID{dataIDX} = squeeze(DataAverage(:,KyIDX,KzIDX,1,1,1,1,1,rowIDX,1,1));
                else
                    
                    listdata.MetaboliteFID{dataIDX} = squeeze(DataAverage(:,KyIDX,KzIDX,1,1,1,1,1,rowIDX,1,1));
                    
                end
            
    end 
end

end

