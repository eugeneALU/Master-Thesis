function metadata = dicominfoAT(filename, varargin)
%DICOMINFO  Read metadata from DICOM message.
%   INFO = DICOMINFO(FILENAME) reads the metadata from the compliant
%   DICOM file specified in the string FILENAME.
%
%   INFO = DICOMINFO(FILENAME, 'dictionary', D) uses the data dictionary
%   file given in the string D to read the DICOM message.  The file in D
%   must be on the MATLAB search path.  The default value is dicom-dict.mat.
%
%   Example:
%
%     info = dicominfo('CT-MONO2-16-ankle.dcm');
%
%   See also DICOMDICT, DICOMREAD, DICOMWRITE, DICOMUID.

%   Copyright 1993-2011 The MathWorks, Inc.
%   $Revision: 1.1.6.11 $  $Date: 2011/11/09 16:50:06 $
%   140811 Added uigetfiel for nargin == 0

% Parse input arguments.
if (nargin < 1)
    
    try
        
        wd = pwd;
        
        cd('/Volumes/data_radiofys/Data/MR/Anders/')
        [FileName,PathName] = uigetfile('*.*','Choose a DICOMDIR file');
        cd(wd)
        filename = fullfile(PathName,FileName);
        
    catch
        error(message('images:dicominfo:tooFewInputs'))
    end
end

% Set the dictionary.
args = parseInputs(varargin{:});
dicomdict('set_current', args.Dictionary)
dictionary = dicomdict('get_current');

% Get the metadata.
try
    
    % Get details about the file to read.
    fileDetails = getFileDetails(filename);
  
    % Ensure the file is actually DICOM.
    if (~isdicom(fileDetails.name))
        error(message('images:dicominfo:notDICOM'))
    end

    % Parse the DICOM file.
    attrs = dicomparse(fileDetails.name, ...
                       fileDetails.bytes, ...
                       getMachineEndian, ...
                       false, ...
                       dictionary); 

    % Process the raw attributes.
    [metadata,attrNames] = processMetadata(attrs, true, dictionary);
    metadata = dicom_set_imfinfo_values(metadata);
    metadata = setMoreImfinfoValues(metadata, fileDetails);
    metadata = processOverlays(metadata, dictionary);
    metadata = processCurveData(metadata, attrs, attrNames);
    
catch ME
    
    dicomdict('reset_current');
    rethrow(ME)
    
end

% Reset the dictionary.
dicomdict('reset_current');



function [metadata,attrNames] = processMetadata(attrs, isTopLevel, dictionary)

if (isempty(attrs))
    metadata = [];
    return
end

% Create a structure for the output and get the names of attributes.
[metadata, attrNames] = createMetadataStruct(attrs, isTopLevel, dictionary);

% Fill the metadata structure, converting data along the way.
for currentAttr = 1:numel(attrNames)
  
    this = attrs(currentAttr);
    metadata.(attrNames{currentAttr}) = convertRawAttr(this, dictionary);
    
end



function processedAttr = convertRawAttr(rawAttr, dictionary)

% Information about whether to swap is contained in the attribute.
swap = needToSwap(rawAttr);

% Determine the correct output encoding.
if (isempty(rawAttr.VR))

    % Look up VR for implicit VR files.  Use 'UN' for unknown
    % tags.  (See PS 3.5 Sec. 6.2.2.)
    vr = findVRFromTag(rawAttr.Group, rawAttr.Element, dictionary);
    if (~isempty(vr))

        % Some attributes have a conditional VR.  Pick the first.
        rawAttr.VR = vr;
        if (numel(rawAttr.VR) > 2)
          rawAttr.VR = rawAttr.VR(1:2);
        end
        
    else
        rawAttr.VR = 'UN';
    end
    
end

% Convert raw data.  (See PS 3.5 Sec. 6.2 for full VR details.)
switch (rawAttr.VR)
case  {'AE','AS','CS','DA','DT','LO','LT','SH','ST','TM','UI','UT'}

    processedAttr = deblankAndStripNulls(char(rawAttr.Data));
    
case {'AT'}
    
    % For historical reasons don't transpose AT.
    processedAttr = dicom_typecast(rawAttr.Data, 'uint16', swap);
    
case {'DS', 'IS'}
 
    processedAttr = sscanf(char(rawAttr.Data), '%f\\');
    
case {'FL', 'OF'}
     
    processedAttr = dicom_typecast(rawAttr.Data, 'single', swap)';
     
case 'FD'
     
    processedAttr = dicom_typecast(rawAttr.Data, 'double', swap)';
    
case 'OB'

    processedAttr = rawAttr.Data';
    
case {'OW', 'US'}
    
    processedAttr = dicom_typecast(rawAttr.Data, 'uint16', swap)';
    
case 'PN'
  
    processedAttr = parsePerson(deblankAndStripNulls(char(rawAttr.Data)));
    
case 'SL'
     
    processedAttr = dicom_typecast(rawAttr.Data, 'int32', swap)';
    
case 'SQ'

    processedAttr = parseSequence(rawAttr.Data, dictionary);

case 'SS'
    
    processedAttr = dicom_typecast(rawAttr.Data, 'int16', swap)';
    
case 'UL'
    
    processedAttr = dicom_typecast(rawAttr.Data, 'uint32', swap)';
    
case 'UN'

    % It's possible that the attribute contains a private sequence
    % with implicit VR; in which case the Data field contains the
    % parsed sequence.
    if (isstruct(rawAttr.Data))
        processedAttr = parseSequence(rawAttr.Data, dictionary);
    else
        processedAttr = rawAttr.Data';
    end

otherwise
    
    % PS 3.5-1999 Sec. 6.2 indicates that all unknown VRs can be
    % interpretted as UN.  
    processedAttr = rawAttr.Data';

end

% Change empty arrays to 0-by-0.
if isempty(processedAttr)
  processedAttr = reshape(processedAttr, [0 0]);
end
  


function byteOrder = getMachineEndian

persistent endian

if (~isempty(endian))
  byteOrder = endian;
  return
end

[~, ~, endian] = computer;
byteOrder = endian;



function args = parseInputs(varargin)

% Set default values
args.Dictionary = dicomdict('get');

% Parse arguments based on their number.
if (nargin > 1)
    
    paramStrings = {'dictionary'};
    
    % For each pair
    for k = 1:2:length(varargin)
        param = lower(varargin{k});
        
             
        if (~ischar(param))
            error(message('images:dicominfo:parameterNameNotString'));
        end
 
        idx = strmatch(param, paramStrings);
        
        if (isempty(idx))
            error(message('images:dicominfo:unrecognizedParameterName', param));
        elseif (length(idx) > 1)
            error(message('images:dicominfo:ambiguousParameterName', param));
        end
    
        switch (paramStrings{idx})
        case 'dictionary'

            if (k == length(varargin))
                error(message('images:dicominfo:missingDictionary'));
            else
                args.Dictionary = varargin{k + 1};
            end
 
        end  % switch
       
    end  % for
           
end



function personName = parsePerson(personString)
%PARSEPERSON  Get the various parts of a person name

% A description and examples of PN values is in PS 3.5-2000 Table 6.2-1.

pnParts = {'FamilyName'
           'GivenName'
           'MiddleName'
           'NamePrefix'
           'NameSuffix'};

if (isempty(personString))
    personName = makePerson(pnParts);
    return
end

people = tokenize(personString, '\\');  % Must quote '\' for calls to STRREAD.

personName = struct([]);

for p = 1:length(people)

    % ASCII, ideographic, and phonetic characters are separated by '='.
    components = tokenize(people{p}, '=');
    
    if (isempty(components))
        personName = makePerson(pnParts);
        return   
    end
        
    
    % Only use ASCII parts.
    
    if (~isempty(components{1}))
        
        % Get the separate parts of the person's name from the component.
        componentParts = tokenize(components{1}, '^');

        % The DICOM standard requires that PN values have five or fewer
        % values separated by "^".  Some vendors produce files with more
        % than these person name parts.
        if (numel(componentParts) <= 5)

            % If there are the correct numbers, put them in separate fields.
            for q = 1:length(componentParts)
                
                personName(p).(pnParts{q}) = componentParts{q};
                
            end
            
        else
            
            % If there are more, just return the whole string.
            personName(p).FamilyName = people{p};
            
        end
        
    else
        
        % Use full string as value if no ASCII is present.
        if (~isempty(components))
            personName(p).FamilyName = people{p};
        end
    
    end
    
end



function personStruct = makePerson(pnParts)
%MAKEPERSON  Make an empty struct containing the PN fields.

for p = 1:numel(pnParts)
    personStruct.(pnParts{p}) = '';
end



function processedStruct = parseSequence(attrs, dictionary)

numItems = countItems(attrs);
itemNames = getItemNames(numItems);

% Initialize the structure to contain this structure.
structInitializer = cat(1, itemNames, cell(1, numItems));
processedStruct = struct(structInitializer{:});

% Process each item (but not delimiters).
item = 0;
for idx = 1:numel(attrs)
  
    this = attrs(idx);
    if (~isDelimiter(this))
        item = item + 1;
        processedStruct.(itemNames{item}) = processMetadata(this.Data, false, dictionary);
    end
    
end



function header = getImfinfoFields

header = {'Filename',      ''
          'FileModDate',   ''
          'FileSize',      []
          'Format',        'DICOM'
          'FormatVersion', 3.0
          'Width',         []
          'Height',        []
          'BitDepth',      []
          'ColorType',     ''}';



function metadata = setMoreImfinfoValues(metadata, d)

metadata.Filename    = d.name;
metadata.FileModDate = d.date;
metadata.FileSize    = d.bytes;



function details = getFileDetails(filename)

% Get the fully qualified path to the file.
fid = fopen(filename);
if (fid > 0)
    
    fullPathFilename = fopen(fid);
    fclose(fid);
    
else
    
    % Look for the file with a different extension.
    file = dicom_create_file_struct;
    file.Filename = filename;
    file = dicom_get_msg(file);
    
    if (isempty(file.Filename))
        error(message('images:dicominfo:noFileOrMessagesFound', filename));
    end
    
    % dicom_get_msg looks up the fully qualified path.
    fullPathFilename = file.Filename;
    
end

details = dir(fullPathFilename);
details.name = fullPathFilename;



function [metadata, attrNames] = createMetadataStruct(attrs, isTopLevel, dictionary)

% Get the attribute names.
totalAttrs = numel(attrs);
attrNames = cell(1, totalAttrs);

for currentAttr = 1:totalAttrs
    attrNames{currentAttr} = ...
        dicomlookup_actions(attrs(currentAttr).Group, ...
                            attrs(currentAttr).Element, ...
                            dictionary);

    % Empty attributes indicate that a public/retired attribute was
    % not found in the data dictionary.  This used to be an error
    % condition, but is easily resolved by providing a special
    % attribute name.
    if (isempty(attrNames{currentAttr}))
        attrNames{currentAttr} = sprintf('Unknown_%04X_%04X', ...
                                         attrs(currentAttr).Group, ...
                                         attrs(currentAttr).Element);
    end
end

% Remove duplicate attribute names.  Keep the last appearance of the attribute.
[tmp, reorderIdx] = unique(attrNames);
if (numel(tmp) ~= totalAttrs)
    warning(message('images:dicominfo:attrWithSameName', 'This DICOM file contains multiple values with the same name.', 'The last appearance is kept.'))
end

uniqueAttrNames = attrNames(sort(reorderIdx));
uniqueTotalAttrs = numel(uniqueAttrNames);

% Create a metadata structure to hold the parsed attributes.  Use a
% cell array initializer, which has a populated section for IMFINFO
% data and an unitialized section for the attributes from the DICOM
% file.
if (isTopLevel)
    structInitializer = cat(2, getImfinfoFields(), ...
                            cat(1, uniqueAttrNames, cell(1, uniqueTotalAttrs)));
else
    structInitializer = cat(1, uniqueAttrNames, cell(1, uniqueTotalAttrs));
end

metadata = struct(structInitializer{:});



function str = deblankAndStripNulls(str)
%DEBLANKANDDENULL  Deblank a string, treating char(0) as a blank.

if (isempty(str))
    return
end

while (~isempty(str) && (str(end) == 0))
    str(end) = '';
end

str = deblank(str);



function vr = findVRFromTag(group, element, dictionary)

% Look up the attribute.
attr = dicomlookup_helper(group, element, dictionary);

% Get the vr.
if (~isempty(attr))
  
    vr = attr.VR;
    
else

    % Private creator attributes should be treated as CS.
    if ((rem(group, 2) == 1) && (element == 0))
        vr = 'UL';
    elseif ((rem(group, 2) == 1) && (element < 256))
        vr = 'CS';
    else
        vr = 'UN';
    end
    
end



function out = processOverlays(in, dictionary)

out = in;

% Look for overlays.
allFields = fieldnames(in);
idx = strmatch('OverlayData', allFields);

if (isempty(idx))
    return
end

% Convert each overlay data attribute.
for p = 1:numel(idx)

    olName = allFields{idx(p)};
    
    % The overlay fields can be present but empty.
    if (isempty(in.(olName)))
        continue;
    end

    % Which repeating group is this?
    [group, element] = dicomlookup_actions(olName, dictionary);

    % Get relevant details.  All overlays are in groups 6000 - 60FE.
    overlay.Rows    = double(in.(dicomlookup_actions(group, '0010', dictionary)));
    overlay.Columns = double(in.(dicomlookup_actions(group, '0011', dictionary)));
    
    sppTag = dicomlookup_actions(group, '0012', dictionary);
    if (isfield(in, sppTag))
        overlay.SamplesPerPixel = double(in.(sppTag));
    else
        overlay.SamplesPerPixel = 1;
    end
    
    bitsTag = dicomlookup_actions(group, '0100', dictionary);
    if (isfield(in, bitsTag))
        overlay.BitsAllocated = double(in.(bitsTag));
    else
        overlay.BitsAllocated = 1;
    end
    
    numTag = dicomlookup_actions(group, '0015', dictionary);
    if (isfield(in, numTag))
        overlay.NumberOfFrames = double(in.(numTag));
    else
        overlay.NumberOfFrames = 1;
    end
    
    % We could potential support more overlays later.
    if ((overlay.BitsAllocated > 1) || (overlay.SamplesPerPixel > 1))
      
        warning(message('images:dicominfo:unsupportedOverlay', sprintf( '(%04X,%04X)', group, element )));
        continue;
        
    end

    % Process the overlay.
    for frame = 1:(overlay.NumberOfFrames)

        overlayData = tobits(in.(olName));
        numSamples = overlay.Columns * overlay.Rows * overlay.NumberOfFrames;
        out.(olName) = permute(reshape(overlayData(1:numSamples), ...
                                       overlay.Columns, ...
                                       overlay.Rows, ...
                                       overlay.NumberOfFrames), ...
                               [2 1 3]);
        
    end
    
end



function out = processCurveData(in, attrs, attrNames)
% Reference - PS 3.3 - 2003 C 10.2.  The Curve Data's final data type
% depends on DataValueRepresentation. Process those attributes here.

% Passing in attrs because we may need to swap the data depending on the
% endianness of the machine and the attribute.

% default
out = in;

% Look for Curve Data.
allFields = fieldnames(in);
idx = strmatch('CurveData', allFields);

if (isempty(idx))
    return
end

% All the Curve Data attributes will have the same endianness, so we just need
% to check one attribute and apply that setting to the rest.

curveDataName = allFields{idx(1)};
curveDataLoc = strncmp(curveDataName,attrNames, length(curveDataName));
swap = needToSwap(attrs(curveDataLoc));

for p = 1 : numel(idx)

    curveDataName = allFields{idx(p)};
    
    underscore = strfind(curveDataName,'_');
    % The data type of the Curve Data comes from the
    % DataValueRepresentation attribute.
    numOfRepeatedAttr = curveDataName(underscore+1:end);
    dvrName = strcat('DataValueRepresentation','_',numOfRepeatedAttr);
        
    if ~isfield(in,dvrName)
        % do nothing
        continue;
    else
        
        dataType = in.(dvrName);
        % See PS 3.3-2003, C 10.2.1.2
        switch dataType
            case 0
                expDataType = 'uint16';
            case 1
                expDataType = 'int16';
            case 2
                expDataType = 'single';
            case 3
                expDataType = 'double';
            case 4
                expDataType = 'int32';
            otherwise
                warning(message('images:dicominfo:unknownDataType', dataType));
        end
        
        % We need to undo any previous swapping before calling typecast, and do the
        % final swapping afterwards.
        
        if swap
            out.(curveDataName) = ...
                swapbytes(typecast(swapbytes(in.(curveDataName)), ...
                                              expDataType));
        else
            out.(curveDataName) = typecast(in.(curveDataName), expDataType);
        end
    end
end
        


function itemNames = getItemNames(numberOfItems)

% Create a cell array of item names, which can be quickly used.
persistent namesCell
if (isempty(namesCell))
    namesCell = generateItemNames(50);
end

% If the number of cached names is too small, expand it and recache.
if (numberOfItems > numel(namesCell))
    namesCell = generateItemNames(numberOfItems);
end

% Return the first n item names.
itemNames = namesCell(1:numberOfItems);



function namesCell = generateItemNames(numberOfItems)

namesCell = cell(1, numberOfItems);
for idx = 1:numberOfItems
    namesCell{idx} = sprintf('Item_%d', idx);
end



function tf = needToSwap(currentAttr)

switch (getMachineEndian)
case 'L'
    if (currentAttr.IsLittleEndian)
        tf = false;
    else
        tf = true;
    end
    
case 'B'
    if (currentAttr.IsLittleEndian)
        tf = true;
    else
        tf = false;
    end
  
otherwise
    error(message('images:dicominfo:unknownEndian', getMachineEndian))

end
    


function tf = isDelimiter(attr)

% True if (FFFE,E00D) or (FFFE,E0DD).
tf = (attr.Group == 65534) && ...
     ((attr.Element == 57357) || (attr.Element == 57565));



function count = countItems(attrs)

if (isempty(attrs))
    count = 0;
else
    % Find the items (FFFE,E000) in the array of attributes (all of
    % which are item tags or delimiters; no normal attributes
    % appear in attrs here). 
    idx = find(([attrs(:).Group] == 65534) & ...
               ([attrs(:).Element] == 57344));
    count = numel(idx);
end
