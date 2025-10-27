function [prof_y] = B20_MovAvg(prof_x,prof_y,wind)

% Applies the moving average filter on a given profile
% By Daniel Cantero Lauer, May 2008

% Copyright (c) 2019, Daniel Cantero. All rights reserved.
% Use, copy or distribution of this program is only allowed with the 
% explicit authorization from the author. Please contact daniel.cantero@ntnu.no

% -------------------------------------------------------------------------
% % ---- Inputs ----
% prof_x = X coordenates of the profile (m)
% prof_y = Matrix containing Y coordenates of the profiles in rows (m)
% wind = Window. Distance to be averaged, should be an even number, and in (m)
%       Recomendation: Use for wind dx multiplied by an EVEN number
% % ---- Outputs ----
% prof_y = New smoother profile
% -------------------------------------------------------------------------

dx = prof_x(2) - prof_x(1);
index = round(wind/dx,0);
index2 = index/2;
prof_ys = prof_y*0;

% Paths loop
for i = 1:size(prof_y,1)
    aux1(i,:) = [ prof_y(i,1)*ones(1,index2),prof_y(i,:),prof_y(i,end)*ones(1,index2)];
end % for i = 1:size(prof_y,1)

aux1 = aux1/(index+1);

% Newest version (6/03/09)
prof_ys(:,1) = sum(aux1(:,1:index+1),2);
aux1 = cumsum(aux1,2);
for i = 2:size(prof_ys,2)
    prof_ys(:,i) = aux1(:,index+i) - aux1(:,i-1);
end % for i = 2:size(prof_ys,2)

% % -- Graphical Check --
% figure; plot(prof_y','k'); hold on; plot(prof_ys','b'); axis tight;

% Alternative: Matlab's command
%%%prof_y = smooth(prof_y,index)';      % Matlab's command

% Output generation
prof_y = prof_ys;

% ---- End of script ----