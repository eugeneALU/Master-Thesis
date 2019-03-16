function CSVcell = getCSVcell(CSVline,CommaPos)

CommaIDX = strfind(CSVline,',');

if CommaPos(1) == 0
   CSVcell = CSVline(1:CommaIDX(1));
elseif CommaPos(2) == -1
   CSVcell = CSVline(CommaIDX(CommaPos(1))+1:end);
    
else
   CSVcell = CSVline(CommaIDX(CommaPos(1))+1 : CommaIDX(CommaPos(2))-1) ;
end



