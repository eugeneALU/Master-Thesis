function [data,info] = readListData(name,varargin)

%%READLISTDATA     Reads a Philips .LIST/.DATA file pair
%
% [DATA,INFO] = READLISTDATA(FILENAME)
%
%   FILENAME is a string containing a file prefix or name of the .LIST
%   header file or .DATA binary file, e.g. RAW_001 or RAW_001.LIST or RAW_001.DATA
%
%   DATA is an N-dimensional array holding the raw binary data.
%
%   INFO is a structure containing details from the .LIST header file
%
% Example:
%  [data,info] = readListData('raw_777.list');
%
%  See also: LOADLABRAW
%
%  Dependencies: none
%

%%Revision History
% * 2006.01.01    initial version - brianwelch


% Allow user to select a file if input FILENAME is not provided or is empty
if nargin < 1 || isempty(name)
  [fn, pn] = uigetfile({'*.list'},'Select a .list file');
  if fn ~= 0
    name = sprintf('%s%s',pn,fn);
  else
    disp('readListData cancelled');
    return;
  end
end

toks = regexp(name,'^(.*?)(\.list|\.data)?$','tokens');
prefix = toks{1}{1};
listname = sprintf('%s.list',prefix);
dataname = sprintf('%s.data',prefix);

fid = fopen(listname,'r');
if fid~=-1,
    listtext = fread(fid,inf,'uint8=>char')';
    fclose(fid);
else
    error( sprintf('cannot open %s for reading', listname) );
end

attr = {'mix','dyn','card','echo','loca','chan','extr1','extr2','ky','kz','n.a.','aver','sign','rf','grad','enc','rtop','rr','size','offset'};

% clean attr names (will be used as variablenames and fieldnames)
for k=1:length(attr),
    attr{k} = cleanFieldname( attr{k} );
end

pattern = '(?<typ>\w+)';
for k=1:length(attr),
    pattern = sprintf('%s\\s+(?<%s>-?\\d+)',pattern,attr{k});
end

info = regexp(listtext, pattern, 'names');

idxSTD = find( strcmp({info(:).typ},'STD') );

for k=1:length(attr),
    eval( sprintf('%s = sort( str2num( char( unique( {info(idxSTD).%s} ) ) ) );',attr{k},attr{k}) );
end

% DATA is a multi-dimensional array organized as
order = {'kx','ky','kz','loca','dyn','card','echo','mix','aver','chan'};
tmpstr = '';
for k=1:length(order),
    if strcmp(order{k},'kx')==1,
        tmpstr = sprintf('%s max(size)/4/2',tmpstr);
    else
        %tmpstr = sprintf('%s length(%s)',tmpstr,order{k});
        tmpstr = sprintf('%s max(%s)-min(%s)+1',tmpstr,order{k},order{k});
    end
end
eval( sprintf('data = zeros([%s]);', tmpstr) );

for k=1:length(order),
    if strcmp(order{k},'kx')==0,
        eval( sprintf('tmp = 1 - min(%s(:));',order{k}) );
        eval( sprintf('%s(:) = %s(:) + tmp;',order{k},order{k}) );
        for j=1:length(idxSTD),
            info(idxSTD(j)).(order{k}) = str2double( info(idxSTD(j)).(order{k}) ) + tmp;
        end
    end
end

fid = fopen(dataname,'r','ieee-le');
if fid==-1,
    error( sprintf('cannot open %s for reading', listname) );
end

hwait = waitbar(0,'=========================================================================================');
set( get( findobj(hwait,'type','axes'),'Title') ,'Interpreter','none');
set( get( findobj(hwait,'type','axes'),'Title') ,'String',sprintf('Reading raw data from %s ...', dataname) );

N = length(idxSTD);
for n=1:N,
    if( fseek(fid, str2num( info(idxSTD(n)).offset ) ,'bof') == 0),
        tmpdata = fread(fid, str2num( info(idxSTD(n)).size )/4 ,'float32');
        tmpdata = tmpdata(1:2:end) + i*tmpdata(2:2:end);
        tmpdata = tmpdata * str2num( info(idxSTD(n)).sign );
        
        tmpstr='';
        for k=1:length(order),
                if strcmp(order{k},'kx')==1,
                    tmpstr = sprintf('%s,1:%d', tmpstr, length(tmpdata));
                else
                    %eval( sprintf('idx = find( %s==str2num(info(idxSTD(n)).%s) );', order{k}, order{k} ) );
                    idx = info(idxSTD(n)).(order{k});
                    tmpstr = sprintf('%s,%d', tmpstr, idx);
                end        
        end
                
        tmpstr(1)=[]; % Delete initial comma
        eval( sprintf('data(%s) = tmpdata;', tmpstr) );
        
    else
        error('Cannot FSEEK to offset=%d in data file %s', info(idxSTD(k)).offset,dataname); 
    end
    if mod(n,100)==99,
        waitbar(n/N,hwait);
    end
end
fclose(fid);
close(hwait);

% # Complex data vector types:
% # --------------------------
% # STD = Standard data vector (image data or spectroscopy data)
% # REJ = Rejected standard data vector
% #       (only for scans with arrhythmia rejection)
% # PHX = Correction data vector for EPI/GraSE phase correction
% # FRX = Correction data vector for frequency spectrum correction
% # NOI = Preparation phase data vector for noise determination
% # NAV = Phase navigator data vector
% #
% # Other attributes of complex data vectors:
% # -----------------------------------------
% # mix    = mixed sequence number
% # dyn    = dynamic scan number
% # card   = cardiac phase number
% # echo   = echo number
% # loca   = location number
% # chan   = synco channel number
% # extr1  = extra attribute 1 (semantics depend on type of scan)
% # extr2  = extra attribute 2 (   ''       ''   ''  ''  ''  '' )
% # kx,ky  = k-space coordinates in 1st and 2nd preparation direction (spectroscopy data)
% # ky,kz  = k-space coordinates in 1st and 2nd preparation direction (image data)
% # aver   = sequence number of this signal average
% # sign   = sign of measurement gradient used for this data vector (1 = positive, -1 = negative)
% # rf     = sequence number of this rf echo (only for TSE, TFE, GraSE)
% # grad   = sequence number of this gradient echo (only for EPI/GraSE)
% # enc    = encoding time (only for EPI/GraSE)
% # rtop   = R-top offset in ms
% # rr     = RR-interval length in ms
% # size   = data vector size   in number of bytes (1 complex element = 2 floats = 8 bytes)
% # offset = data vector offset in number of bytes (first data vector starts at offset 0)
% #
% # The complex data vectors are represented as binary data in little endian single precision IEEE float format.
% #
% # Please note that complex data vector attributes which are not relevant for a certain type of vector
% # may have arbitrary values!

% # Identifying attributes of complex data vectors:
% # -----------------------------------------------
% # The next table specifies the identifying attributes for each type of complex data vector:
% #
% # typ mix   dyn   card  echo  loca  chan  extr1 extr2 ky    kz    aver  sign  rf    grad  enc   rtop  rr    size   offset
% # --- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ------ ------
% #
% # STD   *     *     *     *     *     *     *     *     *     *     *     *     *     *
% # REJ   *     *     *     *     *     *     *     *     *     *     *     *     *     *
% # PHX   *                 *     *     *                                   *     *     *
% # FRX   *                 *     *     *                                   *            
% # NOI                           *     *                                                
% # NAV   *     *     *     *     *     *     *     *     *     *     *     *     *     *

function [s] = cleanFieldname(s)

illegal_chars = {...
    '+','-','*','.',...
    '^','\','/','.',...
    '=','~','<','>',...
    '&','|',':',';',...
    '(',')','{','}',...
    '[',']','{','}',...
    '''','%',' ','!', ...
    '@','#','$','`',...
    '?',',','"',...
    };

general_replacement_char = '_';
firstchar_replacement_char = 'x'; % cannot be an underscore

for k=1:length(illegal_chars),
    s = strrep(s,illegal_chars{k},general_replacement_char);
end

% first character cannot be a number
firstchar_code = double(s(1));
if ( (firstchar_code>=double('0')) & (firstchar_code<=double('9')) )
    s(1) = firstchar_replacement_char;
end

% first character cannot be an underscore
if(s(1)=='_'),
    s(1) = firstchar_replacement_char;
end
