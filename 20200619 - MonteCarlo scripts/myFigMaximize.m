function [] = myFigMaximize(varargin)

% Maximize the figure 
% There are two alternatives
%   1) Based on Java handle. This does the same as clicking on the maximize button in windows
%   2) Changing the outer position of the figure

% Both alternatives found in:
% "http://stackoverflow.com/questions/15286458/automatically-maximize-figure-in-matlab"

% -------------------------------------------------------------------------
% ---- Input ----
% option = (Optional) the method to maximize the window
%   Default is option 1
% ---- Output ----
%
% -------------------------------------------------------------------------

% Input processing
if ~isempty(varargin)
    option = varargin{1};
else
    option = 1;
end % if ~isempty(varargin)

% **** Option 1 **** Java handle
if option == 1
    drawnow;
    warnStruct = warning;
    warning off;
    set(get(handle(gcf),'JavaFrame'),'Maximized',1);
    warning(warnStruct);

% % **** Option 2 **** Outer position
elseif option == 2
    set(gcf,'units','normalized','outerposition',[0 0 1 1]);

end % if option == 1

% ---- End of script ----