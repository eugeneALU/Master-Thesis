function SVS = getLiverTableCSV(TableFile)
%TableFile
fid=fopen(TableFile,'r');
    
[table, COUNT]=fscanf(fid,'%c'); 
fclose(fid);

Metabolite = getvalue('$$CONC','Metabolite','$$MISC',table,'text');

[SVS.L16L09L13Conc SVS.L16L09L13SD SVS.L16L09L13RelConc]=getMetabValue('L16+L09+L13',Metabolite);
[SVS.Lip16Lip13Conc SVS.Lip16Lip13SD SVS.Lip16Lip13RelConc]=getMetabValue('Lip16+Lip13',Metabolite);
[SVS.Lip28L23L21Conc SVS.Lip28L23L21SD SVS.Lip28L23L21RelConc]=getMetabValue('L28+L23+L21',Metabolite);
[SVS.Lip13Conc SVS.Lip13SD SVS.Lip13RelConc]=getMetabValue('Lip13',Metabolite);
[SVS.Lip09Conc SVS.Lip09SD SVS.Lip09RelConc]=getMetabValue('Lip09',Metabolite);
[SVS.Lip16Conc SVS.Lip16SD SVS.Lip16RelConc]=getMetabValue('Lip16',Metabolite);
[SVS.Lip21Conc SVS.Lip21SD SVS.Lip21RelConc]=getMetabValue('Lip21',Metabolite);
[SVS.Lip23Conc SVS.Lip23SD SVS.Lip23RelConc]=getMetabValue('Lip23',Metabolite);
[SVS.Lip28Conc SVS.Lip28SD SVS.Lip28RelConc]=getMetabValue('Lip28',Metabolite);
[SVS.Lip53Lip52Conc SVS.Lip53Lip52SD SVS.Lip53Lip52RelConc]=getMetabValue('Lip53+Lip52',Metabolite);
[SVS.Lip43Conc SVS.Lip43SD SVS.Lip43RelConc]=getMetabValue('Lip43',Metabolite);
[SVS.Lip53Conc SVS.Lip53SD SVS.Lip53RelConc]=getMetabValue('Lip53',Metabolite);
[SVS.Lip52Conc SVS.Lip52SD SVS.Lip52RelConc]=getMetabValue('Lip52',Metabolite);
[SVS.WaterConc SVS.WaterSD SVS.WaterRelConc]=getMetabValue('Water',Metabolite);

% Read lines as string

%$$MISC
s = sprintf('\n');

SVS.FWHM = getvalue('$$MISC','FWHM = ', 'ppm', table,'num');
SVS.SNR = getvalue('$$MISC', 'S/N = ' ,s, table,'num');



end

function [Conc SD RelConc] = getMetabValue(met,str1)
try
% Function to get metabolit vaules for met in str1 

MetIdx2=strfind(str1,[' ' met])-1;
if ~isempty(MetIdx2)
    
MetIdx1=MetIdx2-7;
RelConc=str2double(str1(MetIdx1:MetIdx2));

SDIdx2=MetIdx2-9;
SDIdx1=MetIdx2-11;

SD=str2double(str1(SDIdx1:SDIdx2));
RelConcIdx2=MetIdx2-13;
RelConcIdx1=MetIdx2-20;

Conc=str2double(str1(RelConcIdx1:RelConcIdx2));

else
    Conc=['Can not find ' met ' in LCM table'];
    SD='NA';
    RelConc='NA';
end

catch
    Conc=['Error finding ' met ' in LCM table'];
    SD='NA';
    RelConc='NA';
end

end

