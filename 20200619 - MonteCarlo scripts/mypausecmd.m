%function [] = mypausecmd
% Function to pause the execution of a script anytime, and resume whenever wanted

% Daniel Cantero Lauer - September 2008 
%   (Revised November 2012)
%   Revised January 2014 = Not a function, removed unnecessary variable, recycle on

% -------------------------------------------------------------------------
% To use this comand:
% Include this command within a loop of the script
% To pause the simulation:
%   1) While the script is working create a file in the same directory
%      with the name "pause".
%   2) To resume the calculations follow the instructions.
%      Type on the command window "return".
% Additional comments
% The created file will be removed automatically.
% There is no significant increase in computational time because of this
% command. It takes less than a milisecon to run each time.
% -------------------------------------------------------------------------

%pausecmd = dir('pause');
%if size(pausecmd,1) > 0
if size(dir('pause'),1) > 0
    recycle on;
    delete('pause');
    disp('To continue the execution please:');
    disp('Type "return" on the command window (or "dbquit" to stop simulation)');
    keyboard
end % if 

% ---- End of script ----