function [CommaStart CommaEnd] = getCommaPos(CSVtitleLine,Str)

% getCommaPos function for wich comma seperates a 
%
% Orginal getValueCSV written by Anders Tisell 20110112
% Change to getCommaPos 20111103 by Anders Tisell


% Find index for comma
CommaVecTitle = strfind(CSVtitleLine,',');
CommaIdxTitle = strfind(CSVtitleLine,Str);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Find string in title

if length(CommaIdxTitle) > 1
    
    if ~isempty(strfind(CSVtitleLine, [',' Str ',']))
        
        CommaIdxTitle = strfind(CSVtitleLine, [',' Str ',']);
        
    elseif ~isempty(strfind(CSVtitleLine, [' ' Str ',']))
        
        CommaIdxTitle = strfind(CSVtitleLine, [' ' Str ',']);
        
    elseif ~isempty(strfind(CSVtitleLine, [',' Str ' ']))
        
        CommaIdxTitle = strfind(CSVtitleLine, [',' Str ' ']);
        
    elseif ~isempty(strfind(CSVtitleLine, [' ' Str ' ']))
        
        CommaIdxTitle = strfind(CSVtitleLine, [' ' Str ' ']);
        
    end
    
end

if isempty(CommaIdxTitle) 
    
    CommaIdxCSVlineStart = []; 
    CommaIdxCSVlineEnd = [];
    

else
   
    
    CommaStart = length(find(CommaVecTitle <= CommaIdxTitle));
    CommaEnd = CommaStart + 1;
 %   findStartComma(CommaIdxTitle,CommaVecTitle);
  %  CommaEnd = findEndComma(CommaIdxTitle,CommaVecTitle);
    
    
end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Sub function for finding the comma in the CSV line that are infront of
% the wanted value.

function CommaIDXout = findStartComma(CommaIdxIn,CommaVecTitle)

if ~isempty(find(CommaVecTitle <= CommaIdxIn))
    
    CommaIDXout = max(CommaVecTitle(CommaVecTitle <= CommaIdxIn));

else
    
    CommaIDXout = 0;

end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Sub function for finding the comma in the CSV line that are after the
% wanted value.

function CommaIDXout = findEndComma(CommaIdxIn,CommaVecTitle)

if ~isempty(find(CommaVecTitle > CommaIdxIn))
    
    CommaIDXout = min(CommaVecTitle(CommaVecTitle > CommaIdxIn));

else
    
    CommaIDXout = -1;

end

end
