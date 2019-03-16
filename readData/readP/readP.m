function [ Pdata ] = readP( pfile )
%readP Reads GE sepctroscopy P files in to a matlab struct Pdata
%   Detailed explanation goes here
%
%   Version 1 2014-04-17
%   Writen by Anders Tisell, Linkoping University, Sweden
%   Based on the code plotP by Fred Frigo 
%   

if(nargin == 0)
    wd=pwd;
    cd('/Volumes/data_radiofys/Data/MR/SpectroGE/');
    [fname, pname] = uigetfile('*.*', 'Select Pfile');
    
    cd(wd)
    pfile = strcat(pname, fname);
end

% Open Pfile to read reference scan data.
fid = fopen(pfile,'r', 'ieee-le');
if fid == -1
    err_msg = sprintf('Unable to locate Pfile %s', pfile)
    return;
end

status = fseek(fid, 0, 'bof');

[f_hdr_value, count] = fread(fid, 1, 'real*4'); % Read the first 32 bits of the file to get the version number. 

Pdata.version = f_hdr_value;
status = fseek(fid, 1468, 'bof');
Pdata.header_size = fread(fid,1,'integer*4');


% Read header information
status = fseek(fid, 0, 'bof');
[hdr_value, count] = fread(fid, 102, 'integer*2');
Pdata.npasses = hdr_value(33);
Pdata.nslices = hdr_value(35);
Pdata.nechoes = hdr_value(36);
Pdata.nframes = hdr_value(38);
Pdata.point_size = hdr_value(42);
Pdata.da_xres = hdr_value(52);
Pdata.da_yres = hdr_value(53);
Pdata.rc_xres = hdr_value(54);
Pdata.rc_yres = hdr_value(55);
Pdata.start_recv = hdr_value(101);
Pdata.stop_recv = hdr_value(102);
Pdata.nreceivers = (Pdata.stop_recv - Pdata.start_recv) + 1;


status = fseek(fid, 0, 'bof');
% float    rdb_hdr_rdbm_rev;
[hdr_value, count] = fread(fid, 1, 'real*4');

% int      rdb_hdr_run_int;           /* Rdy pkt Run Number */
[hdr_value, count] = fread(fid, 1, 'real*4');

% short    rdb_hdr_scan_seq;          /* Rdy pkt Sequence Number */
% char     rdb_hdr_run_char [6];      /* Rdy pkt Run no in char */
% char     rdb_hdr_scan_date [10];    /*  */
% char     rdb_hdr_scan_time [8];     /*  */
% char     rdb_hdr_logo [10];         /* rdbm  used to verify file */
% 
% short    rdb_hdr_file_contents;     /* Data type 0=emp 1=nrec 2=rw   0, 1, 2 */
% short    rdb_hdr_lock_mode;         /* unused */
% short    rdb_hdr_dacq_ctrl;         /* rhdacqctrl bit mask       15 bits */
% short    rdb_hdr_recon_ctrl;        /* rhrcctrl bit mask         15 bits */
% unsigned short    rdb_hdr_exec_ctrl;         /* rhexecctrl bit mask       15 bits */
% short    rdb_hdr_scan_type;         /* bit mask          15 bits */
% short    rdb_hdr_data_collect_type; /* rhtype  bit mask      15 bits */
% short    rdb_hdr_data_format;       /* rhformat  bit mask        15 bits */
% short    rdb_hdr_recon;             /* rhrecon proc-a-son recon  0 - 100 */
%     short    rdb_hdr_datacq;            /* rhdatacq proc-a-son dacq */
% 
%     short    rdb_hdr_npasses;           /* rhnpasses  passes for a scan  0 - 256 */
%     short    rdb_hdr_npomp;             /* rhnpomp  pomp group slices    1,2 */
%     unsigned short    rdb_hdr_nslices;  /* rhnslices  slices in a pass   0 - 256 */
%     short    rdb_hdr_nechoes;           /* rhnecho  echoes of a slice    1 - 32 */
%     short    rdb_hdr_navs;              /* rhnavs  num of excitiations   1 - 32727 */
%     short    rdb_hdr_nframes;           /* rhnframes  yres       0 - 1024 */
%     short    rdb_hdr_baseline_views;    /* rhbline  baselines        0 - 1028 */
%     short    rdb_hdr_hnover;            /* rhhnover  overscans       0 - 1024 */
%     unsigned short  rdb_hdr_frame_size; /* rhfrsize  xres        0 - 32768 */
%     short    rdb_hdr_point_size;        /* rhptsize          2 - 4 */
% 
%     short    rdb_hdr_vquant;            /* rhvquant 3d volumes       1 */
% 
%     short    rdb_hdr_cheart;            /* RX Cine heart phases      1 - 32 */
%     float    rdb_hdr_ctr;               /* RX Cine TR in sec     0 - 3.40282e38*/
%     float    rdb_hdr_ctrr;              /* RX Cine RR in sec         0 - 30.0 */
% 
%     short    rdb_hdr_initpass;          /* rhinitpass allocate passes    0 - 32767 */
%     short    rdb_hdr_incrpass;          /* rhincrpass tps autopauses 0 - 32767 */
% 
%     short    rdb_hdr_method_ctrl;       /* rhmethod  0=recon, 1=psd  0, 1 */
%     unsigned short    rdb_hdr_da_xres;  /* rhdaxres          0 - 32768 */
%     short    rdb_hdr_da_yres;           /* rhdayres          0 - 2049 */
%     short    rdb_hdr_rc_xres;           /* rhrcxres          0 - 1024 */
%     short    rdb_hdr_rc_yres;           /* rhrcyres          0 - 1024 */
%     short    rdb_hdr_im_size;           /* rhimsize          0 - 512 */
%     int      rdb_hdr_rc_zres;           /* power of 2 > rhnslices    0 - 128 */