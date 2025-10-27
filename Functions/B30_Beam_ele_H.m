function He = B30_Beam_ele_H(L,E,I)

% Generation of beam element Stress-Displacement matrix

% Copyright (c) 2019, Daniel Cantero. All rights reserved.
% Use, copy or distribution of this program is only allowed with the 
% explicit authorization from the author. Please contact daniel.cantero@ntnu.no

% % ---------------------------------------------------------------
% % ---- Input ----
% L = Length of Beam element
% E = Young's Modulus of the beam element
% I = Moment of inertia of the beam element
% % ---- Output ----
% He = Element Stress-Displacement matrix
% % ---------------------------------------------------------------

He = E*I * ...
    [[ -6/L^2, -4/L,  6/L^2, -2/L];...
    [  6/L^2,  2/L, -6/L^2,  4/L]];

% ---- End of function ----