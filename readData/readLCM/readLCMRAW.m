function raw = readLCMRAW(rawFile)
% Function for reading the raw files produced by bin2raw script proveded by
% Stephen. 
% Funtion implemented 150610

fid=fopen(rawFile,'r');

[rawText, COUNT]=fscanf(fid,'%c'); 

fclose(fid);

 = getvalue('$$CONC','Metabolite','$$MISC',table,'text');
raw.