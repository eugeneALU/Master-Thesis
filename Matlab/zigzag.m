function seq = zigzag(SI)
%
%  Description:
%  ------------
%  This function is used to build the corresponding sequences of a given
%  scaled gray level image matrix from 45' degree direction. The whole process is using zigzag method
%  It can handle nonsquare image matrix
%
% Author:
% -------
%    (C)Xunkai Wei <xunkai.wei@gmail.com>
%    Beijing Aeronautical Technology Research Center
%    Beijing %9203-12,10076
%
% History:
%  -------
% Creation: beta  Date: 01/11/2007
% Revision: 1.0   Date: 12/11/2007
% 
% Trick: all the sequence starts or ends lie on the boundary.

% initializing the variables
%----------------------------------
c = 1; % initialize colum indicator
r = 1; % initialize row   indicator

rmin = 1; % row   boundary checker
cmin = 1; % colum boundary checker

rmax = size(SI, 1); % get row numbers
cmax = size(SI, 2); % get colum numbers

i = 1; % counter for current ith element
j = 1; % indicator for determining sequence interval

% intialize sequence mark
sq_up_begin=1;
sq_down_begin=1;

% Output contain value
output = zeros(1, rmax * cmax);
%----------------------------------

while ((r <= rmax) && (c <= cmax))

    % for current point, judge its zigzag direction up 45, or down 45, or
    % 0,or down 90

    %% up 45 direction
    if (mod(c + r, 2) == 0)
        %%  if we currently walk to the first row
        if (r == rmin)
            % First, record current point
            output(i) = SI(r, c);
            % if we walk to right last colum
            if (c == cmax)
                % go down to another row and end of this sequence
                % This next point is the begin point of next sequence
                r = r + 1;
            else
                % Continue to move to next column and end of this sequence
                % This next point is the begin point of next sequence
                c = c + 1;
            end
            sq_up_end = i;
            sq_down_begin = i+1;
            seq{j}=output(sq_up_begin:sq_up_end);
            j = j + 1;
            % add counter
            i = i + 1;
            
        %% if we currently walk to the last column
        elseif ((c == cmax) && (r < rmax))
            % first record the point
            output(i) = SI(r, c);
            % then move straight down to next row
            r = r + 1;
            sq_up_end = i;
            sq_down_begin =i+1;
            seq{j}=output(sq_up_begin:sq_up_end);
            j=j+1;
            % add counter
            i = i + 1;
            
        %% all other cases i.e. nonboundary points
        elseif ((r > rmin) && (c < cmax))
            output(i) = SI(r, c);
            % move to next up 45 point
            r = r - 1;
            c = c + 1;
            i = i + 1;
        end
    %% down 45 direction
    else
        %% if we walk to the last row
        if ((r == rmax) && (c <= cmax))
            % firstly record current point
            output(i) = SI(r, c);
            % move right to next point
            c = c + 1;
            sq_down_end = i;
            seq{j}=output(sq_down_begin:sq_down_end);
            sq_up_begin =i+1;
            j = j + 1;
            % add counter
            i = i + 1;
        %% if we walk to the first column
        elseif (c == cmin)
            %first record current point
            output(i) = SI(r, c);
            %if we run to last row
            if (r == rmax)
                % move to next column and end of this sequence
                c = c + 1;
            else
                % move to next row and end of this sequence
                r = r + 1;
            end
            sq_down_end = i;
            sq_up_begin =i+1;
            seq{j}=output(sq_down_begin:sq_down_end);
            j = j + 1;
            i = i + 1;
            
        % all other cases without boundary point
        elseif ((r < rmax) && (c > cmin))
            output(i) = SI(r, c);
            r = r + 1;
            c = c - 1;
            i = i + 1;
        end

    end

    %% bottom right element(the last one)
    if ((r == rmax) && (c == cmax))       
        output(i) = SI(r, c);
        sq_end = i;
        seq{j}=output(sq_end);
        break
    end
end