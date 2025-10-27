function [varargout] = mysubplot(num_rows,num_cols,sub_num,varargin)

% Alternative version of subplot() command
% The difference are that:
%   - the subplots are located on the figure leaving no gaps between them
%   - It can be used for plotting more subplots than num_rows x num_cols. It
%     generates then a new figure
%   - Optional use matlab's subplot() command (no change of margins). Simply
%     define the optional input as something different than a 1x4 array

% The inputs of subplot() and mysubplot() are identical

% -------------------------------------------------------------------------
% ---- Input ----
% num_rows = Number of subplot rows that the figure has
% num_cols = Number of subplot cols that the figure has
% sub_num = Current subplot number
% -- Optional input --
% margins = Two options:
%   a) 1x4 vector containing the margins for the current subplot
%      [x_left,x_right,y_bottom,y_top]
%   b) Something different than a 1x4 array. Then the standard subplot() command
%      is used, and the margins are not modified.
% ---- Output ----
% h = subplot handle
% -------------------------------------------------------------------------

%h = subplot(num_rows,num_cols,sub_num);
%h = subplot(num_rows,num_cols,sub_num,'HandleVisibility','off');

% Default values
change_margins = 1;     % Change margin flag

% Starting new figure (if necessary)
if sub_num > num_rows*num_cols
    if mod(sub_num,num_rows*num_cols) == 1
        figure;
    end % if mod(sub_num,num_rows*num_cols) == 1
end % if sub_num > num_rows*num_cols

sub_num = mymod(sub_num,num_rows*num_cols);

% Input processing
margins = zeros(1,4);
if nargin > 3
    if length(varargin{1}) == 4
        margins = varargin{1};
    else
        change_margins = 0;
    end % if length(varargin{1}) == 4
end % if nargin > 3

if change_margins == 1
    
    % Current subplot column number
    col_num = mymod(sub_num,num_cols);

    % Current subplot row number;
    row_num = ceil(sub_num/num_cols);

    x_left = margins(1)/num_cols;
    x_right = margins(2)/num_cols;
    y_bottom = margins(3)/num_rows;
    y_top = margins(4)/num_rows;

    % New definition of the subplot's axis position
    if length(col_num) == 1
        % 1-cell subplot
        Position_vector = [(1/num_cols)*(col_num-1), (1/num_rows)*(num_rows-row_num), 1/num_cols 1/num_rows];
    else
        % Multiple-cells subplot
        Position_vector = [(1/num_cols)*(min(col_num)-1), (1/num_rows)*(num_rows-max(row_num)), length(unique(col_num))/num_cols, length(unique(row_num))/num_rows];
    end % if length(col_num) == 1
    Position_vector = Position_vector + [x_left,y_bottom,-(x_left+x_right),-(y_bottom+y_top)];
    h = axes('Position',Position_vector);
    
elseif change_margins == 0
    
    subplot(num_rows,num_cols,sub_num);
    
end % if change_margins == 1

% Generating output
if nargout > 0
    varargout{1} = h;
end % if nargout > 0

% ---- End of script ----