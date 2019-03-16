function tableCSVline = getTableCSVingenia(TableFile)
%TableFile
fid=fopen(TableFile,'r');
    
[table, COUNT]=fscanf(fid,'%c'); 
fclose(fid);

Metabolite = getvalue('$$CONC','Metabolite','$$MISC',table,'text');

[SVS.AcnConc SVS.AcnSD SVS.AcnRelConc]=getMetabValue('Acn',Metabolite);
[SVS.ActConc SVS.ActSD SVS.ActRelConc]=getMetabValue('Act',Metabolite);
[SVS.AlaConc SVS.AlaSD SVS.AlaRelConc]=getMetabValue('Ala',Metabolite);
[SVS.AspConc SVS.AspSD SVS.AspRelConc]=getMetabValue('Asp',Metabolite);
[SVS.CrConc SVS.CrSD SVS.CrRelConc]=getMetabValue('Cr',Metabolite);
[SVS.PCrConc SVS.PCrSD SVS.PCrRelConc]=getMetabValue('PCr',Metabolite);
[SVS.tCrConc SVS.tCrSD SVS.tCrRelConc]=getMetabValue('Cr+PCr',Metabolite);
[SVS.GABAConc SVS.GABASD SVS.GABARelConc]=getMetabValue('GABA',Metabolite);
[SVS.GlcConc SVS.GlcSD SVS.GlcRelConc]=getMetabValue('Glc',Metabolite);
[SVS.GlnConc SVS.GlnSD SVS.GlnRelConc]=getMetabValue('Gln',Metabolite);
[SVS.GluConc SVS.GluSD SVS.GluRelConc]=getMetabValue('Glu',Metabolite);
[SVS.GPCConc SVS.GPCSD SVS.GPCRelConc]=getMetabValue('GPC',Metabolite);
[SVS.InsConc SVS.InsSD SVS.InsRelConc]=getMetabValue('Ins',Metabolite);
[SVS.LacConc SVS.LacSD SVS.LacRelConc]=getMetabValue('Lac',Metabolite);
[SVS.NAAConc SVS.NAASD SVS.NAARelConc]=getMetabValue('NAA',Metabolite);
[SVS.NAAGConc SVS.NAAGSD SVS.NAAGRelConc]=getMetabValue('NAAG',Metabolite);
[SVS.PChConc SVS.PChSD SVS.PChRelConc]=getMetabValue('PCh',Metabolite);
[SVS.PyrConc SVS.PyrSD SVS.PyrRelConc]=getMetabValue('Pyr',Metabolite);
[SVS.ScylloConc SVS.ScylloSD SVS.ScylloRelConc]=getMetabValue('Scyllo',Metabolite);
[SVS.SucConc SVS.SucSD SVS.SucRelConc]=getMetabValue('Suc',Metabolite);
[SVS.TauConc SVS.TauSD SVS.TauRelConc]=getMetabValue('Tau',Metabolite);
[SVS.CrCH2Conc SVS.CrCH2SD SVS.CrCH2RelConc]=getMetabValue('-CrCH2',Metabolite);
[SVS.GuaConc SVS.GuaSD SVS.GuaRelConc]=getMetabValue('Gua',Metabolite);
[SVS.GPCPChConc SVS.GPCPChSD SVS.GPCPChRelConc]=getMetabValue('GPC+PCh',Metabolite);
[SVS.NAANAAGConc SVS.NAANAAGSD SVS.NAANAAGRelConc]=getMetabValue('NAA+NAAG',Metabolite);
[SVS.GluGlnConc SVS.GluGlnSD SVS.GluGlnRelConc]=getMetabValue('Glu+Gln',Metabolite);
[SVS.Lip13aConc SVS.Lip13aSD SVS.Lip13aRelConc]=getMetabValue('Lip13a',Metabolite);
[SVS.Lip13bConc SVS.Lip13bSD SVS.Lip13bRelConc]=getMetabValue('Lip13b',Metabolite);
[SVS.Lip09Conc SVS.Lip09SD SVS.Lip09RelConc]=getMetabValue('Lip09',Metabolite);
[SVS.MM09Conc SVS.MM09SD SVS.MM09RelConc]=getMetabValue('MM09',Metabolite);
[SVS.Lip20Conc SVS.Lip20SD SVS.Lip20RelConc]=getMetabValue('Lip20',Metabolite);
[SVS.MM20Conc SVS.MM20SD SVS.MM20RelConc]=getMetabValue('MM20',Metabolite);
[SVS.MM12Conc SVS.MM12SD SVS.MM12RelConc]=getMetabValue('MM12',Metabolite);
[SVS.MM14Conc SVS.MM14SD SVS.MM14RelConc]=getMetabValue('MM14',Metabolite);
[SVS.MM17Conc SVS.MM17SD SVS.MM17RelConc]=getMetabValue('MM17',Metabolite);
[SVS.Lip13aLip13bConc SVS.Lip13aLip13bSD SVS.Lip13aLip13bRelConc]=getMetabValue('Lip13a+Lip13b',Metabolite);
[SVS.MM14Lip13aLip13bMM12Conc SVS.MM14Lip13aLip13bMM12SD SVS.MM14Lip13aLip13bMM12RelConc]=getMetabValue('MM14+Lip13a+Lip13b+MM12',Metabolite);
[SVS.MM09Lip09Conc SVS.MM09Lip09SD SVS.MM09Lip09RelConc]=getMetabValue('MM09+Lip09',Metabolite);
[SVS.MM20Lip20Conc SVS.MM20Lip20SD SVS.MM20Lip20RelConc]=getMetabValue('MM20+Lip20',Metabolite);
% Read lines as string

%$$MISC
s = sprintf('\n');

SVS.FWHM = getvalue('$$MISC','FWHM = ', 'ppm', table,'num');
SVS.SNR = getvalue('$$MISC', 'S/N = ' ,s, table,'num');

tableCSVline = [num2str(SVS.tCrConc) ' , ' num2str(SVS.tCrConc * SVS.tCrSD / 100) ' , ' ...
    num2str(SVS.InsConc) ' , ' num2str(SVS.InsConc * SVS.InsSD / 100) ' , ' ...
    num2str(SVS.NAAConc) ' , ' num2str(SVS.NAAConc * SVS.NAASD / 100) ' , ' ...
    num2str(SVS.NAAGConc) ' , ' num2str(SVS.NAAGConc * SVS.NAAGSD / 100) ' , ' ...
    num2str(SVS.NAANAAGConc) ' , ' num2str(SVS.NAANAAGConc * SVS.NAANAAGSD / 100) ' , ' ...
    num2str(SVS.GPCPChConc) ' , ' num2str(SVS.GPCPChConc * SVS.GPCPChSD / 100) ' , ' ...
    num2str(SVS.LacConc) ' , ' num2str(SVS.LacConc * SVS.LacSD / 100) ' , ' ...
    num2str(SVS.GluConc) ' , ' num2str(SVS.GluConc * SVS.GluSD / 100) ' , ' ...
    num2str(SVS.GlnConc) ' , ' num2str(SVS.GlnConc * SVS.GlnSD / 100) ' , ' ...
    num2str(SVS.GluGlnConc) ' , ' num2str(SVS.GluGlnConc * SVS.GluGlnSD / 100) ];
  
end

function [Conc SD RelConc] = getMetabValue(met,str1)

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

end

