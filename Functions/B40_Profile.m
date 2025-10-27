function [Calc] = B40_Profile(Calc)

% Gathers together all the functions related to the Profile:
%   Loading of a saved profile
%   Calculation of new profile
%   Saving of generated profile

% Copyright (c) 2019, Daniel Cantero. All rights reserved.
% Use, copy or distribution of this program is only allowed with the 
% explicit authorization from the author. Please contact daniel.cantero@ntnu.no

% -------------------------------------------------------------------------
% ---- Input ----
% Calc = Structure with Calculation variables, including at least:
% ---- Output ----
% Calc = Addition of new fields to the structure
% -------------------------------------------------------------------------

if Calc.Profile.Load.on == 0
    
    % -- Profile Calculation --
    [Calc] = B19_RoadProfile(Calc);

    % -- Saving Profile --
    B38_Profile_Save(Calc);
    
elseif Calc.Profile.Load.on == 1

    % -- Loading Pofile --
    [Calc] = B39_Profile_Load(Calc);

end % if Calc.Profile.Load.on == 0

% ---- End of function ----