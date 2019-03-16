function [ doubleVAXG ] = uint64le_to_VAXG(uint32le)
%UINT64LE_TO_VAXG Converts from IEEE-LE (UINT32) to VAXG (double)
%  This function takes a vector columns of raw 32bit unsigned integers
%  (little endian), combines them into 64bit unsigned integers, and
%  converts the result into the equivalent floating point numbers using
%  the specification for the VAXG floating point number format.
%  VAXD and VAXG are two 64bit floating point formats. VAXF is the 32bit
%  floating point format used in conjunction with both VAXD and VAXG.
%  http://www.opengroup.org/onlinepubs/9629399/chap14.htm#tagfcjh_20
%
%  The function is intended to be called from FREADVAXG.
%
%  See also UINT32LE_TO_VAXF, UINT64LE_TO_VAXD, FREADVAXD, FREADVAXG

%   Copyright 2009 The MathWorks, Inc.

%% Define floating value properties for VAX architecture
% The generic equation for a floating point number is:
% (-1)^double(S) * (F+C) * A^(double(E)-B);
% Different operating systems and file formats utilize different values
% for A, B, and C. F, E, and S are computed from the appropriate bits in
% the number as stored on disk.

    A = 2   ;%VAX specific
    B = 1024;%VAX specific
    C = 0.5 ;%VAX specific

%% Convert raw unsigned number into right answer
% VAX (64bit)      <--WORD1--><--WORD2--><--WORD3--><--WORD4-->
% IEEE-LE (32bit)  <--WORD2--><--WORD1-->
% IEEE-LE (32bit)  <--WORD4--><--WORD3-->

    uint32le  = reshape(uint32le,2,[])';
    uint32leA = uint32le(:,1);
    uint32leB = uint32le(:,2);

% Swap WORD1 and WORD2

    word2   = bitshift(bitshift(uint32leA,  0), -16);%mask FFFF0000
    word1   = bitshift(bitshift(uint32leA, 16), -16);%mask 0000FFFF
    vaxIntA = bitor(bitshift(word1,16), bitshift(word2, 0));
    
% Swap WORD3 and WORD4

    word4   = bitshift(bitshift(uint32leB,  0), -16);%mask FFFF0000
    word3   = bitshift(bitshift(uint32leB, 16), -16);%mask 0000FFFF
    vaxIntB = bitor(bitshift(word3,16), bitshift(word4, 0));

% Create UINT64: <--WORD1--><--WORD2--><--WORD3--><--WORD4-->

    vaxIntA = bitshift(uint64(vaxIntA),32);
    vaxIntB = uint64(vaxIntB);
    vaxInt  = bitor(vaxIntA,vaxIntB);
       
% Pull out the sign, exponent, and fractional component
% VAX FLOAT BYTES  <-----WORD1----><-----WORD2----><-----WORD1----><-----WORD2---->
% VAX FLOAT BITS   0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF
% Sign Exp Fract   SEEEEEEEEEEEFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF

    S  = bitshift(vaxInt, -63);
    E  = bitshift(bitshift(vaxInt, 1),-53);
    F  = bitshift(bitshift(vaxInt,12),-12);

% Construct the floating point number from SEF (Sign, Exp, Fract)
% http://www.codeproject.com/KB/applications/libnumber.aspx
% http://www.opengroup.org/onlinepubs/9629399/chap14.htm#tagfcjh_20

    M = C+double(F)./9007199254740992; %VAX Specific 90071992547409924=2^53
    doubleVAXG = (-1).^double(S) .* M .* A.^(double(E)-B);%Generic

end
%#eml
