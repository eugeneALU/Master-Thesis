function value = getValueCSV(CSVtitleLine,CSVline,Str)

% getValueCSV function for finding values corresponding to the csv title line
%
% value = getValueCSV(CSVtitleLine,CSVline,'Str')
%
% Writen by Anders Tisell 20110112


% Find index for where comma ar
CommaVecTitle =strfind(CSVtitleLine,',');
CommaVecCSVline =strfind(CSVline,',');
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
    
    value = [];
    
%elseif length(CommaVecTitle) ~= length(CommaVecCSVline)
%    
%    try
%    warning(['Title line and CSV line do not containe the same number of commma. For: ' CSVline])
%   
%    value = [];
%    catch
%        warning('Title line and CSV line do not containe the same number of commma')
%    value = [];
%    end
%    
%    
%    %error('Title line and CSV line do not containe the same number of commma')

else
    
    
    CommaStartIdxTitle = findStartComma(CommaIdxTitle,CommaVecTitle);
    
    if CommaStartIdxTitle > 0
        
        CommaIdxCSVlineStart = CommaVecCSVline(find(CommaVecTitle == CommaStartIdxTitle))+1;
    else
        CommaIdxCSVlineStart = 1;
    end
    
    
    CommaEndIdxTitle = findEndComma(CommaIdxTitle,CommaVecTitle);
    
    if CommaEndIdxTitle > 0
        CommaIdxCSVlineEnd = CommaVecCSVline(find(CommaVecTitle == CommaEndIdxTitle))-1;
    else
        CommaIdxCSVlineEnd = length(CSVline);
    end
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Remove whitespace between , and the value
    while ~isempty(strfind(CSVline(CommaIdxCSVlineStart),' '))
     %   disp('Remove white space')
        CommaIdxCSVlineStart = CommaIdxCSVlineStart + 1;
    end
    
    while ~isempty(strfind(CSVline(CommaIdxCSVlineEnd),' '))
    %    disp('Remove white space')
        CommaIdxCSVlineEnd =  CommaIdxCSVlineEnd - 1;
    end
    
    value = CSVline(CommaIdxCSVlineStart:CommaIdxCSVlineEnd);
    
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
    
    CommaIDXout = 0;

end

end
