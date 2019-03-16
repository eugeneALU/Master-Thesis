function [WM GM CSF Non] =  readSegTable(Table)


fid = fopen(Table);
    [A,COUNT] = fscanf(fid,'%c');
    fclose(fid);

    
    s = sprintf('\n');
    
TotalLine = getvalue('Slice	WM	GM	CSF	NON','Sum',s,A,'text');


WM = str2num(TotalLine(1:3));
GM = str2num(TotalLine(5:7));
CSF =str2num(TotalLine(9:11));
if CSF > 100
    Non = str2num(TotalLine(13:end));
else 
    Non = str2num(TotalLine(12:end));
end


