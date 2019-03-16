function value = getvalue(text,starttext,endtext,file,format)

% HELP text
% value = getvalue(text,starttext,endtext,file)
% 
%
% IN variabels:
% file, is a text string variable read 
% (eg. fid = fopen(Parfile); [file,COUNT] = fscanf(fid,'%c'); fclose(fid); 
% 
% text, is a key text string that should be found in file, the first 
% occurance is used if the string present more than once string occurance in file.
% 
% endtext, a string which specifies the endpoint of the readout.
%
% format, text for text tring output, num for numerical array
% output
%
% OUT variabels:
% 
% Call functions:
%


% Function developed 2005-10-11, by Olof Dahlqvist
% Revised 2008-04.14, Anders Tisell 
% 

% ------------Initialization----------------
value = [];
% ------------------------------------------

try
    pos=findstr(text,file);
    if isempty(pos)
        error(['Could not find '  text ' in parameter file']);
    end
    
    pos=findstr(starttext,file(pos(1):end))+pos(1);
    if isempty(pos)
        error(['Could not find '  starttext ' after ' text ' in parameter file']);
    end
    pos=pos(1)+length(starttext);
    pos2=findstr(endtext,file(pos:end))+pos;
    if isempty(pos2)
        error(['Could not find '  endtext ' after ' text ' in parameter file']);
    end
    pos2=pos2-2;
    if strcmp(format,'text')
        value=strtrim(file(pos:pos2));
    elseif strcmp(format,'num')
        value=str2num(file(pos:pos2));
    end
catch
    str=lasterr;
    value = ['undefined, ' str];
end
end