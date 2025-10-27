function HSe = B32_Beam_ele_HS(L,E,I)

% Generation of beam element Shear Stress-Displacement matrix

% Copyright (c) 2019, Daniel Cantero. All rights reserved.
% Use, copy or distribution of this program is only allowed with the 
% explicit authorization from the author. Please contact daniel.cantero@ntnu.no

% % ---------------------------------------------------------------
% % ---- Input ----
% L = Length of Beam element
% E = Young's Modulus of the beam element
% I = Moment of inertia of the beam element
% % ---- Output ----
% HSe = Element Stress-Displacement matrix
% % ---------------------------------------------------------------

% HSe = E*I * ...
%     [[  12/L^3,  6/L^2, -12/L^3,  6/L^2];...
%     [ -12/L^3, -6/L^2,  12/L^3, -6/L^2]];

HSe = E*I * ...
    [[  12/L^3,  6/L^2, -12/L^3,  6/L^2];...
    [ 12/L^3, 6/L^2,  -12/L^3, 6/L^2]];     % Sign changed!

% Sign changed because in the calculation of shear all the contributions to
% shear in a node are averaged, but because of the sign it gives the sum of
% two quantities, one positive and one negative, giving a (almost) zero
% average.

% ---- End of function ----