function [data,info] = loadLABRAW(filename,varargin)
%%LOADLABRAW     Load a Philips LABRAW file
%
% [DATA,INFO] = LOADLABRAW(FILENAME)
%
%   FILENAME is a string containing a file prefix or name of the LAB
%   hexadecimal label file or RAW data file, e.g. RAW_001 or RAW_001.LAB or RAW_001.RAW
%
%   DATA is an N-dimensional array holding the raw k-space data.
%
%   INFO is a structure containing details from the LAB hexadecimal label file
%
% [DATA,INFO] = LOADLABRAW([])
%
%   When the passed FILENAME is not provided or is an empty array or empty
%   string.  The user chooses a file using UIGETFILE.
%
% [DATA,INFO] = LOADLABRAW(FILENAME,'OptionName1',OptionValue1,...)
%
%   Options can be passed to LOADLABRAW to control the range/pattern of
%   loaded data, verbose output, etc.  The list below shows the avialable
%   options.  Names are case-sensitive
%
%       OptionName          OptionValue       Description
%       ----------          -----------     ---------------
%       'coil'              numeric         coils
%       'kx'                numeric         k-space kx samples
%       'ky'                numeric         k-space ky rows (E1)
%       'kz'                numeric         k-space kz rows (E2)
%       'e3'                numeric         k-space 3rd encoding dim
%       'loc'               numeric         locations
%       'ec'                numeric         echoes
%       'dyn'               numeric         dynamics
%       'ph'                numeric         cardiac phases
%       'row'               numeric         rows
%       'mix'               numeric         mixes
%       'avg'               numeric         averages
%       'verbose'           logical         [ true |{false}]
%       'savememory'        logical         [{true}| false ]
%       'correct_phase'     logical         [{true}| false ]
%       'correct_dcoffset'  logical         [{true}| false ]
%       'correct_pda'       logical         [{true}| false ]
%
%       When 'savememory' is true, SINGLE precision is used instead of DOUBLE
%
%       When 'correct_phase' is true, random and measurement phase
%       corrections are applied to the data.
%       
%       When 'correct_dcoffset' is true, the DC offset for each coil
%       channel is removed using the FRC noise data.  If no FRC noise data
%       is available, no DC offset is removed. If multiple stacks of FRC
%       noise data is present, a .PDF.XML file is needed to know which
%       locations go into which stack.
%
%       When 'correct_pda' is true, the profile dependent amplification
%       is corrected using information from a .PDF.XML file with the same
%       prefix as the loaded LAB/RAW file pair.  If no .PDF.XML file is 
%       available, no pda correction is applied.
%
%   Example:
%       myfile = 'example.LAB';
%       [data,info] = loadLABRAW(myfile,'coil',[1 5],'verbose',true);
%
% [DATA,INFO] = LOADLABRAW(FILENAME,LOADOPTS)
%
%   LOADOPTS is a structure with fieldnames equal to any of the possible
%   OptionNames.
%
%   Example:
%       loadopts.coil = [1 5];
%       loadopts.verbose = true;
%       [data,info] = loadLABRAW(myfile,loadopts);
%
%   For any dimension, values may be repeated and appear in any order.
%   Values that do not intersect with the available values for that
%   dimension will be ignored.  If the intersection of the user-defined
%   dimension values and the available dimension range has length zero, an
%   error is generated.  The order of the user-defined pattern is preserved.
%
%   Example:
%       % load a specific pattern of locations (-1 will be ignored)
%       loadopts.loc = [1 1 2 1 1 2 -1];
%       [data,info] = loadLABRAW(myfile,loadopts);
%
% INFO = LOADLABRAW(FILENAME)
%
%   If only one return argument is provided, the INFO structure will be
%   returned.  DATA will not be loaded (fast execution).
%
% INFO structure contents
%
%   The INFO structure contains all the information from the LAB file in
%   a filed names LABELS as well as other useful information to describe
%   and to work with the loaded DATA array.  The list below describes some
%   of the additional fields found within INFO
%
%   FieldName              Description
%   ---------              ------------------------------------------------
%   FILENAME               filename of the loaded data
%   LOADOPTS               structure containing the load options (see above)
%   DIMS                   structure containing the DATA dimension names and values
%   LABELS                 structure containing label names and values
%   LABELS_ROW_INDEX_ARRAY (see below)
%   LABEL_FIELDNAMES       names used for the labels
%   IDX                    structure of arrays of index of different label types
%   FSEEK_OFFSETS          byte offsets in the .RAW file for each data vector
%   NLABELS                # of total labels avialable in the LAB file
%   NLOADEDLABELS          # of labels loaded from the LABRAW file
%   NDATALABELS            # of labels in the returned data array (may contain repeats)
%   DATASIZE               array showing the size of the returned DATA array
%   FRC_NOISE_DATA         array of the FRC noise data
%
%   The INFO.LABELS_ROW_INDEX_ARRAY is a special array that is the same
%   size as the DATA array (minus the first two dimensions used to store
%   COIL and KX).  A given index for a given raw data vector in the DATA
%   array will return the label index number describing the details of that
%   raw data vector in the INFO.LABELS array when that same index is used
%   with the INFO.LABELS_ROW_INDEX_ARRAY.  This provides a quick way to
%   recall the details of any individual raw data vector contained within DATA.
%   If the INFO.TABLE_ROW_INDEX_ARRAY holds a ZERO for a given index, there
%   was no label from the LAB file that matched the dimension location in DATA.
%
%  See also: LOADPARREC, LOADPDFXML, LOADCPX
%

% Revision History
% * 2008.11.07    initial version - welcheb
% * 2012.08.03    added dc offset and pda correction - welcheb
% * 2012.08.22    read EPI phase correction data into info structure

% Start execution time clock and initialize DATA and INFO to empty arrays
tic;
data = [];
info = [];

% Initialize INFO structure
% Serves to fix the display order
info.filename = [];
info.loadopts = [];
info.dims = [];
info.labels = [];
info.labels_row_index_array = [];
info.label_fieldnames = [];
info.idx = [];
info.fseek_offsets = [];
info.nLabels = [];
info.nLoadedLabels = [];
info.nDataLabels = [];
info.nNormalDataLabels = [];
info.datasize = [];

% Allow user to select a file if input FILENAME is not provided or is empty
if nargin < 1 || isempty(filename)
  [fn, pn] = uigetfile({'*.raw'},'Select a RAW file');
  if fn ~= 0
    filename = sprintf('%s%s',pn,fn);
  else
    disp('LOADLABRAW cancelled');
    return;
  end
end

% Parse the filename.
% It may be the LAB filename, RAW filename or just the filename prefix
% Instead of REGEXP, use REGEXPI which igores case
toks = regexpi(filename,'^(.*?)(\.lab|\.raw)?$','tokens');
prefix = toks{1}{1};
labname = sprintf('%s.LAB',prefix);
rawname = sprintf('%s.RAW',prefix);
pdfxmlname = sprintf('%s.PDF.XML',prefix);
info.filename = filename;

% Open LAB file and read all hexadecimal labels
labfid = fopen(labname,'r');
if labfid == -1
  error('Cannot open %s for reading', labname);
end

% Read all hexadecimal labels
unparsed_labels = fread (labfid,[16 Inf], 'uint32=>uint32');
info.nLabels = size(unparsed_labels,2);
fclose(labfid);

% Parse hexadecimal labels
% Inspired by Holger Eggers' readRaw.m.  See arsrcglo1.h for more details.
% RKR - converts hexidecimal values from .LAB file into decimal label vals
info.labels.DataSize.vals         = unparsed_labels(1,:);

info.labels.LeadingDummies.vals   = bitshift (bitand(unparsed_labels(2,:), (2^16-1)),  -0);
info.labels.TrailingDummies.vals  = bitshift (bitand(unparsed_labels(2,:), (2^32-1)), -16);

info.labels.SrcCode.vals          = bitshift (bitand(unparsed_labels(3,:), (2^16-1)),  -0);
info.labels.DstCode.vals          = bitshift (bitand(unparsed_labels(3,:), (2^32-1)), -16);

info.labels.SeqNum.vals           = bitshift (bitand(unparsed_labels(4,:), (2^16-1)),  -0);
info.labels.LabelType.vals        = bitshift (bitand(unparsed_labels(4,:), (2^32-1)), -16);

info.labels.ControlType.vals      = bitshift( bitand(unparsed_labels(5,:),  (2^8-1)),  -0);
info.labels.MonitoringFlag.vals   = bitshift( bitand(unparsed_labels(5,:), (2^16-1)),  -8);
info.labels.MeasurementPhase.vals = bitshift( bitand(unparsed_labels(5,:), (2^24-1)), -16);
info.labels.MeasurementSign.vals  = bitshift( bitand(unparsed_labels(5,:), (2^32-1)), -24);

info.labels.GainSetting.vals      = bitshift( bitand(unparsed_labels(6,:),  (2^8-1)),  -0);
info.labels.Spare1.vals           = bitshift( bitand(unparsed_labels(6,:), (2^16-1)),  -8);
info.labels.Spare2.vals           = bitshift (bitand(unparsed_labels(6,:), (2^32-1)), -16);

info.labels.ProgressCnt.vals      = bitshift (bitand(unparsed_labels(7,:), (2^16-1)),  -0);
info.labels.Mix.vals              = bitshift (bitand(unparsed_labels(7,:), (2^32-1)), -16);

info.labels.Dynamic.vals          = bitshift (bitand(unparsed_labels(8,:), (2^16-1)),  -0);
info.labels.CardiacPhase.vals     = bitshift (bitand(unparsed_labels(8,:), (2^32-1)), -16);

info.labels.Echo.vals             = bitshift (bitand(unparsed_labels(9,:), (2^16-1)),  -0);
info.labels.Location.vals         = bitshift (bitand(unparsed_labels(9,:), (2^32-1)), -16);

info.labels.Row.vals              = bitshift (bitand(unparsed_labels(10,:), (2^16-1)),  -0);
info.labels.ExtraAtrr.vals        = bitshift (bitand(unparsed_labels(10,:), (2^32-1)), -16);

info.labels.Measurement.vals      = bitshift (bitand(unparsed_labels(11,:), (2^16-1)),  -0);
info.labels.E1.vals               = bitshift (bitand(unparsed_labels(11,:), (2^32-1)), -16);

info.labels.E2.vals               = bitshift (bitand(unparsed_labels(12,:), (2^16-1)),  -0);
info.labels.E3.vals               = bitshift (bitand(unparsed_labels(12,:), (2^32-1)), -16);

info.labels.RfEcho.vals           = bitshift (bitand(unparsed_labels(13,:), (2^16-1)),  -0);
info.labels.GradEcho.vals         = bitshift (bitand(unparsed_labels(13,:), (2^32-1)), -16);

info.labels.EncTime.vals          = bitshift (bitand(unparsed_labels(14,:), (2^16-1)),  -0);
info.labels.RandomPhase.vals      = bitshift (bitand(unparsed_labels(14,:), (2^32-1)), -16);

info.labels.RRInterval.vals       = bitshift (bitand(unparsed_labels(15,:), (2^16-1)),  -0);
info.labels.RTopOffset.vals       = bitshift (bitand(unparsed_labels(15,:), (2^32-1)), -16);

info.labels.ChannelsActive.vals   = unparsed_labels(16,:);

clear unparsed_labels;

% Find unique values of each label field
info.label_fieldnames = fieldnames(info.labels);
for k = 1:length(info.label_fieldnames)
  info.labels.(info.label_fieldnames{k}).uniq = unique( info.labels.(info.label_fieldnames{k}).vals );
end

% Calculate fseek offsets
info.fseek_offsets = zeros(info.nLabels,1);
info.fseek_offsets(1)=512; % add mysterious 512 byte offset to begin reading file
for k = 2:info.nLabels
  info.fseek_offsets(k) = info.fseek_offsets(k-1)+ info.labels.DataSize.vals(k-1) - info.labels.TrailingDummies.vals(k-1)  - info.labels.LeadingDummies.vals(k-1);
end
info.idx.no_data = find(info.labels.DataSize.vals == 0);
info.fseek_offsets(info.idx.no_data) = -1;

% Find indices of different label control types
% See arsrcglo1.h for more details.
standard_labels = info.labels.LabelType.vals==32513;
info.idx.NORMAL_DATA         = find(info.labels.ControlType.vals== 0 & standard_labels);
info.idx.DC_OFFSET_DATA      = find(info.labels.ControlType.vals== 1 & standard_labels);
info.idx.JUNK_DATA           = find(info.labels.ControlType.vals== 2 & standard_labels);
info.idx.ECHO_PHASE_DATA     = find(info.labels.ControlType.vals== 3 & standard_labels);
info.idx.NO_DATA             = find(info.labels.ControlType.vals== 4 & standard_labels);
info.idx.NEXT_PHASE          = find(info.labels.ControlType.vals== 5 & standard_labels);
info.idx.SUSPEND             = find(info.labels.ControlType.vals== 6 & standard_labels);
info.idx.RESUME              = find(info.labels.ControlType.vals== 7 & standard_labels);
info.idx.TOTAL_END           = find(info.labels.ControlType.vals== 8 & standard_labels);
info.idx.INVALIDATION        = find(info.labels.ControlType.vals== 9 & standard_labels);
info.idx.TYPE_NR_END         = find(info.labels.ControlType.vals==10 & standard_labels);
info.idx.VALIDATION          = find(info.labels.ControlType.vals==11 & standard_labels);
info.idx.NO_OPERATION        = find(info.labels.ControlType.vals==12 & standard_labels);
info.idx.DYN_SCAN_INFO       = find(info.labels.ControlType.vals==13 & standard_labels);
info.idx.SELECTIVE_END       = find(info.labels.ControlType.vals==14 & standard_labels);
info.idx.FRC_CH_DATA         = find(info.labels.ControlType.vals==15 & standard_labels);
info.idx.FRC_NOISE_DATA      = find(info.labels.ControlType.vals==16 & standard_labels);
info.idx.REFERENCE_DATA      = find(info.labels.ControlType.vals==17 & standard_labels);
info.idx.DC_FIXED_DATA       = find(info.labels.ControlType.vals==18 & standard_labels);
info.idx.DNAVIGATOR_DATA     = find(info.labels.ControlType.vals==19 & standard_labels);
info.idx.FLUSH               = find(info.labels.ControlType.vals==20 & standard_labels);
info.idx.RECON_END           = find(info.labels.ControlType.vals==21 & standard_labels);
info.idx.IMAGE_STATUS        = find(info.labels.ControlType.vals==22 & standard_labels);
info.idx.TRACKING            = find(info.labels.ControlType.vals==23 & standard_labels);
info.idx.FLUOROSCOPY_TOGGLE  = find(info.labels.ControlType.vals==24 & standard_labels);
info.idx.REJECTED_DATA       = find(info.labels.ControlType.vals==25 & standard_labels);
info.idx.UNKNOWN27           = find(info.labels.ControlType.vals==27 & standard_labels);
info.idx.UNKNOWN28           = find(info.labels.ControlType.vals==28 & standard_labels);

% number of total labels
info.nLabels = length(info.labels.DataSize.vals);

% Calculate number of standard, normal data labels
info.nNormalDataLabels = length(info.idx.NORMAL_DATA);

% Dimension names
dimnames = {'coil','kx','ky','kz','E3','loc','ec','dyn','ph','row','mix','avg'};
dimfields = {'N/A','N/A','E1','E2','E3','Location','Echo','Dynamic','CardiacPhase','Row','Mix','Measurement'};

% Initialize dimension data to zero
info.dims.nCoils         = 0;
info.dims.nKx            = 0;
info.dims.nKy            = 0;
info.dims.nKz            = 0;
info.dims.nE3            = 0;
info.dims.nLocations     = 0;
info.dims.nEchoes        = 0;
info.dims.nDynamics      = 0;
info.dims.nCardiacPhases = 0;
info.dims.nRows          = 0;
info.dims.nMixes         = 0;
info.dims.nMeasurements  = 0;

% Calculate max number of active coils
maxChannelsActiveMask = 0;
info.ChannelsActiveBitLocations = [];
bit_location = 0;
ChannelsActive_NORMAL_DATA_unique = unique( info.labels.ChannelsActive.vals(info.idx.NORMAL_DATA) );
for k = 1:length(ChannelsActive_NORMAL_DATA_unique)
  maxChannelsActiveMask = bitor(maxChannelsActiveMask,ChannelsActive_NORMAL_DATA_unique(k));
end
while maxChannelsActiveMask > 0
  bit_location = bit_location + 1;
  if bitand(maxChannelsActiveMask, 1),
    info.dims.nCoils = info.dims.nCoils + 1;
    info.ChannelsActiveBitLocations = [info.ChannelsActiveBitLocations bit_location];
  end
  maxChannelsActiveMask = bitshift (maxChannelsActiveMask, -1);
end

% Calculate number of active coils for each label index
% preallocate info.ChannelsActive
info.ChannelsActive(1) = {1:maxChannelsActiveMask};
info.ChannelsActive(2:info.nLabels) = info.ChannelsActive(1);
for k = 1:length(info.labels.ChannelsActive.uniq)
  tmp_label_idx = find( info.labels.ChannelsActive.vals == info.labels.ChannelsActive.uniq(k) );
  ChannelsActiveMask = info.labels.ChannelsActive.vals( tmp_label_idx(1) );
  numChannelsActive = 0;
  bit_location = 0;
  ChannelsActiveIdx = [];
  while ChannelsActiveMask > 0
    bit_location = bit_location + 1;
    if bitand(ChannelsActiveMask, 1)
      numChannelsActive = numChannelsActive + 1;
      ChannelsActiveIdx = [ChannelsActiveIdx find(bit_location==info.ChannelsActiveBitLocations)];
    end
    ChannelsActiveMask = bitshift (ChannelsActiveMask, -1);
  end
  info.numChannelsActive(tmp_label_idx) = numChannelsActive;
  info.ChannelsActiveIdx(tmp_label_idx) = {ChannelsActiveIdx};
end

% Calculate dimensions of normal data
% why not use length(info.labels.XX.uniq)?
info.dims.nKx            = max(info.labels.DataSize.vals(info.idx.NORMAL_DATA)) / info.dims.nCoils / 2 / 2;
info.dims.nKy            = length(unique(info.labels.E1.vals(info.idx.NORMAL_DATA)));
info.dims.nKz            = length(unique(info.labels.E2.vals(info.idx.NORMAL_DATA)));
info.dims.nE3            = length(unique(info.labels.E3.vals(info.idx.NORMAL_DATA)));
info.dims.nLocations     = length(unique(info.labels.Location.vals(info.idx.NORMAL_DATA)));
info.dims.nEchoes        = length(unique(info.labels.Echo.vals(info.idx.NORMAL_DATA)));
info.dims.nDynamics      = length(unique(info.labels.Dynamic.vals(info.idx.NORMAL_DATA)));
info.dims.nCardiacPhases = length(unique(info.labels.CardiacPhase.vals(info.idx.NORMAL_DATA)));
info.dims.nRows          = length(unique(info.labels.Row.vals(info.idx.NORMAL_DATA)));
info.dims.nMixes         = length(unique(info.labels.Mix.vals(info.idx.NORMAL_DATA)));
info.dims.nMeasurements  = length(unique(info.labels.Measurement.vals(info.idx.NORMAL_DATA)));

% With known possible dimension names, the load options can now be parsed
p = inputParser;
p.StructExpand = true;
p.CaseSensitive = true;
p.KeepUnmatched = false; % throw an error for unmatched inputs
p.addRequired('filename', @ischar);
for k = 1:length(dimnames)
  p.addParamValue(dimnames{k}, [], @isnumeric);
end
p.addParamValue('verbose', false, @islogical);
p.addParamValue('savememory', true, @islogical);
p.addParamValue('correct_phase', true, @islogical);
p.addParamValue('correct_dcoffset', true, @islogical);
p.addParamValue('correct_pda', true, @islogical);
p.parse(filename, varargin{:});

% Return loadopts structure inside INFO structure
% remove filename field - it is passed as the first required argument
info.loadopts = rmfield(p.Results,'filename');

% Find the unique set of values for each dimension name
info.dims.coil = 1:info.dims.nCoils;
info.dims.kx   = 1:info.dims.nKx;
for k = 3:length(dimnames) % skip coil and kx
  info.dims.(dimnames{k}) = unique(info.labels.(dimfields{k}).vals(info.idx.NORMAL_DATA));
end

% Find intersection of available dimensions with LOADOPTS dimensions
for k = 1:length(dimnames)
  if ~isempty(info.loadopts.(dimnames{k})),
    info.dims.(dimnames{k}) = intersect_a_with_b(info.loadopts.(dimnames{k}),info.dims.(dimnames{k}));
  end
end

% Calculate data size
info.datasize = cellfun(@(x)length(info.dims.(x)), dimnames);

% throw error if any dimension size is zero
if any(info.datasize == 0)
  error('size of selected data to load has zero length along dimension(s):  ''%s'' ', dimnames{info.datasize==0});
end

% Skip data loading if only one output argument is provided, return INFO
if nargout < 2
  info.labels_row_index_array = 1:size(info.labels,1);
  data = info;
  return;
end

% Create array to hold label row numbers for loaded data
% skip the coil and kx dimensions
info.labels_row_index_array = zeros(info.datasize(3:end));

% Pre-allocate DATA array
if info.loadopts.savememory
  data = zeros(info.datasize,'single');
else
  data = zeros(info.datasize);
end

% Read RAW data for selected dimension ranges
fidraw = fopen(rawname,'r','ieee-le');
if fidraw < 0
  error('cannot open RAW file: %s', rawname);
end
info.nLoadedLabels = 0;

% Read FRC noise data
for k = 1:length(info.idx.FRC_NOISE_DATA),
  frc_noise_idx = info.idx.FRC_NOISE_DATA(k);
  frc_noise_samples_per_coil = info.labels.DataSize.vals(frc_noise_idx) / 2 / 2 / info.numChannelsActive(frc_noise_idx);
  info.FRC_NOISE_DATA{k} = zeros(info.numChannelsActive(frc_noise_idx),frc_noise_samples_per_coil,'single');
  byte_offset = info.fseek_offsets(frc_noise_idx);
  fseek(fidraw, byte_offset, 'bof');
  rawdata_1d = double(fread(fidraw, double(info.labels.DataSize.vals(frc_noise_idx)/2) , 'int16'));
  for sample=1:frc_noise_samples_per_coil,
    for coil=1:info.numChannelsActive(frc_noise_idx),
      re_idx = 2*frc_noise_samples_per_coil*(coil-1) + 2*(sample-1) + 1;
      im_idx = re_idx + 1;
      info.FRC_NOISE_DATA{k}(coil,sample) = rawdata_1d(re_idx) + 1i*rawdata_1d(im_idx);
    end
  end
end

% load PDF.XML file if it is needed and present
if ( ( (info.loadopts.correct_dcoffset) || (info.loadopts.correct_pda) ) && (exist(pdfxmlname)==2) ),
    info.pdfxml = loadPDFXML(pdfxmlname);
end

% Prepare for DC offset correction
if ( (info.loadopts.correct_dcoffset) && (isfield(info,'pdfxml')) ),
    
    info.dcoffset.nStacks = length(info.FRC_NOISE_DATA);
    info.dcoffset.nCoils = length(info.ChannelsActiveBitLocations);
    info.dcoffset.dcoffset_table = complex(zeros(info.dcoffset.nStacks, info.dcoffset.nCoils));
    for stack=1:info.dcoffset.nStacks,
        noise_vectors = info.FRC_NOISE_DATA{stack};
        for coil=1:info.dcoffset.nCoils,
            noise_vector = noise_vectors(coil,:);
            info.dcoffset.dcoffset_table(stack,coil) = mean( noise_vector(:) );
        end
    end
    
    stack_cumsum = cumsum(info.pdfxml.PDF_EXAM_PARS.EX_GEO_stacks_slices);
    loc_indices = sort( unique( info.labels.Location.vals(info.idx.NORMAL_DATA) ) ) + 1; 
    stack_idx = 1;
    for idx=1:length(loc_indices),
        if loc_indices(idx) > stack_cumsum(stack_idx),
            stack_idx = stack_idx + 1;
        end
        info.dcoffset.loc_stack_idx(loc_indices(idx)) = stack_idx;
    end
        
end

% Prepare for PDA correction
if ( (info.loadopts.correct_pda) && (isfield(info,'pdfxml')) ),
    info.pda.AWDEF_MAX_CHANNELS = length( info.pdfxml.PDF_EXAM_PARS.EX_COIL_channel_synthesis);
    info.pda.AWDEF_MAX_GAIN_SETTINGS = length( info.pdfxml.PDF_HARDWARE_PARS.HW_rec_atts ) / info.pda.AWDEF_MAX_CHANNELS;
    info.pda.HW_rec_atts = info.pdfxml.PDF_HARDWARE_PARS.HW_rec_atts;
    info.pda.HW_rec_att_phases = info.pdfxml.PDF_HARDWARE_PARS.HW_rec_att_phases;
    
    info.pda.pda_ampl_factors = complex(zeros(info.pda.AWDEF_MAX_CHANNELS,info.pda.AWDEF_MAX_GAIN_SETTINGS));
    idx=1;
    for channel_idx = 1:info.pda.AWDEF_MAX_CHANNELS,
        for gain_idx = 1:info.pda.AWDEF_MAX_GAIN_SETTINGS,
            att_dB = info.pda.HW_rec_atts(idx);
            phase_degrees = info.pda.HW_rec_att_phases(idx);
            ampl = (10.0).^(-att_dB/20.0);
            phase_radians = phase_degrees * pi / 180.0;
            info.pda.pda_ampl_factors(channel_idx,gain_idx) = cos(-phase_radians)/ampl + 1i * sin(-phase_radians)/ampl;
            idx=idx+1;
        end
    end
end

%for n = 1:length(info.idx.NORMAL_DATA)
for label_idx = info.idx.NORMAL_DATA
  
  load_flag=1;
  dim_assign_indices_full_array = [];
  
  %label_idx = info.idx.NORMAL_DATA(n);
  
  for k = 3:length(dimfields)
    
    dimval = info.labels.(dimfields{k}).vals(label_idx);
    
    % it is allowed that the dimval appears more than once
    % in the requested dimension ranges to be loaded
    dim_assign_indices = find(dimval==info.dims.(dimnames{k}));
    
    if isempty(dim_assign_indices)
      load_flag = 0;
      break;
    else
      
      if k > 3
        
        dim_assign_indices_full_array_new = zeros( size(dim_assign_indices_full_array,1)*length(dim_assign_indices), size(dim_assign_indices_full_array,2)+1);
        
        mod_base_a = size(dim_assign_indices_full_array,1);
        mod_base_b = length(dim_assign_indices);
        
        for d=1:size(dim_assign_indices_full_array_new,1),
          dim_assign_indices_full_array_new(d,:) = ...
            [dim_assign_indices_full_array(mod(d,mod_base_a)+1,:) ...
            dim_assign_indices(mod(d,mod_base_b)+1)];
        end
        
      else
        dim_assign_indices_full_array_new = dim_assign_indices(:);
      end
      
      dim_assign_indices_full_array = dim_assign_indices_full_array_new;
      
    end
  end
  
  if load_flag
    
    info.nLoadedLabels = info.nLoadedLabels + 1;
    
    byte_offset = info.fseek_offsets(label_idx);
    fseek(fidraw, byte_offset, 'bof');
    
    raw_data_fread_size = double(info.numChannelsActive(label_idx) * info.dims.nKx * 2);
    rawdata_1d = double(fread(fidraw, raw_data_fread_size, 'int16'));
    
    % Phase correction
    if info.loadopts.correct_phase,
        RandomPhase = double(info.labels.RandomPhase.vals(label_idx));
        MeasurementPhase = double(info.labels.MeasurementPhase.vals(label_idx));
        c = exp (- 1i * pi * (2 * RandomPhase / (2^16-1) + MeasurementPhase / 2));
    end
    
    % parse into real and imaginary parts for each coil
    ChannelsActiveIdx = info.ChannelsActiveIdx(label_idx);
    coil_assign_idx = ChannelsActiveIdx{:};
    rawdata_2d = complex(zeros(info.dims.nCoils,info.dims.nKx),zeros(info.dims.nCoils,info.dims.nKx));
    for kx = 1:info.dims.nKx
      for coil = 1:info.numChannelsActive(label_idx)
        re_idx = 2*info.dims.nKx*(coil-1) + 2*(kx-1) + 1;
        %im_idx = re_idx + 1;
        rawdata_2d(coil_assign_idx(coil),kx) = c * (rawdata_1d(re_idx) + 1i*rawdata_1d(re_idx+1));
      end
    end
    
    % account for measurement sign
    if info.labels.MeasurementSign.vals(label_idx)
      rawdata_2d = fliplr(rawdata_2d);
    end
    
    % apply dc offset correction
    if ( (info.loadopts.correct_dcoffset) && isfield(info,'dcoffset') ),
        for coil=1:info.numChannelsActive(label_idx),
            coil_idx = coil_assign_idx(coil);
            loc_idx = info.labels.Location.vals(label_idx)+1;
            stack_idx = info.dcoffset.loc_stack_idx(loc_idx);
            dc_offset = info.dcoffset.dcoffset_table(stack_idx,coil_idx);
            rawdata_2d(coil_idx,:) = rawdata_2d(coil_idx,:) - dc_offset;
        end
    end
    
    % apply pda correction
    if ( (info.loadopts.correct_pda) && isfield(info,'pda') ),
        for coil=1:info.numChannelsActive(label_idx),
            coil_idx = coil_assign_idx(coil);
            gain_idx = info.labels.GainSetting.vals(label_idx)+1;
            pda_ampl_factor = info.pda.pda_ampl_factors(coil_idx,gain_idx);
            rawdata_2d(coil_idx,:) = pda_ampl_factor * rawdata_2d(coil_idx,:) ;
        end
    end
    
    % select choosen coils
    rawdata_2d = rawdata_2d(info.dims.coil,:);
    
    % select choosen kx
    rawdata_2d = rawdata_2d(:,info.dims.kx);
    
    % insert rawdata_2d into proper locations of the data array
    for d = 1:size(dim_assign_indices_full_array,1)
      
      dim_assign_str = sprintf(',%d', dim_assign_indices_full_array(d,:) );
      
      % delete initial comma
      dim_assign_str(1) = [];
      
      % assign index to table_index table
      eval( sprintf('info.labels_row_index_array(%s)=%d;', dim_assign_str, label_idx) );
      
      % assign read image to correct location in data array
      eval( sprintf('data(:,:,%s)=rawdata_2d;', dim_assign_str) );
      
    end
    
  end
  
end

% RKR: Read PH data
for k = 1:length(info.idx.ECHO_PHASE_DATA),
  echo_phase_idx = info.idx.ECHO_PHASE_DATA(k);
  echo_phase_samples_per_coil = info.labels.DataSize.vals(echo_phase_idx) / 2 / 2 / info.numChannelsActive(echo_phase_idx);
  info.ECHO_PHASE_DATA{k} = zeros(info.numChannelsActive(echo_phase_idx),echo_phase_samples_per_coil,'single');
  byte_offset = info.fseek_offsets(echo_phase_idx);
  fseek(fidraw, byte_offset, 'bof');
  rawdata_1d = double(fread(fidraw, double(info.labels.DataSize.vals(echo_phase_idx)/2) , 'int16'));
  % Phase correction
    RandomPhase = double(info.labels.RandomPhase.vals(echo_phase_idx));
    MeasurementPhase = double(info.labels.MeasurementPhase.vals(echo_phase_idx));
    c = exp (- 1i * pi * (2 * RandomPhase / (2^16-1) + MeasurementPhase / 2));
  
  for sample=1:echo_phase_samples_per_coil,
    for coil=1:info.numChannelsActive(echo_phase_idx),
      re_idx = 2*echo_phase_samples_per_coil*(coil-1) + 2*(sample-1) + 1;
      im_idx = re_idx + 1;
      info.ECHO_PHASE_DATA{k}(coil,sample) = c * (rawdata_1d(re_idx) + 1i*rawdata_1d(im_idx));
    end
  end
  
  % account for measurement sign
%   if info.labels.MeasurementSign.vals(echo_phase_idx)
%     info.ECHO_PHASE_DATA{k} = fliplr(info.ECHO_PHASE_DATA{k});
%   end
  
end

% Close RAW file
fclose(fidraw);

% Calculate total raw data blobs
size_data = size(data);
max_img_dims = size_data(3:end);
info.nDataLabels = prod(max_img_dims);

% If VERBOSE, display execution information
if info.loadopts.verbose
  fprintf('Loaded %d of %d available normal data labels', info.nLoadedLabels, info.nNormalDataLabels);
  tmpstr = '';
  for k = 1:length(dimnames),
    tmpstr = sprintf('%s, # %s: %d', tmpstr, dimnames{k}, length(info.dims.(dimnames{k})) );
  end
  fprintf('Data contains %d raw labels - %s', info.nDataLabels, tmpstr(3:end));
  fprintf('Total execution time = %.3f seconds', toc);
end


function c = intersect_a_with_b(a,b)
%%INTERSECT_A_WITH_B Find intersection of vector a with vector b without sorting
% possible replacement:  (not tested yet)
% c = b(any(bsxfun(@(x,y) x==y, a, b')));
c = a;
% work backwards in order to use [] assignment
for k = length(a):-1:1,
  if isempty(find(a(k)==b, 1)),
    c(k) = [];
  end
end



% force c to be a row vector for easier display
c = c(:).';