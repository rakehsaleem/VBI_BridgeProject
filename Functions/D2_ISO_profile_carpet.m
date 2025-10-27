function [prof_x,prof_y,Gd]= D2_ISO_profile_carpet(L,Ly,dx,paths,Gd)

% Generation of ISO profile carpet

% % -------------------------------------------------------------------------
% % ---- Input ----
% L = Length of profile on X direcction (m)
% Ly = Length of profile on Y direcction (m) (Should be "even decimal")
% dx = Spatial sampling frequency on X direction
% paths = Vector containing entry points for path calculation (m)
% Gd = Roughness coeficient in ISO form [m^3/cycles]
%       If Gd = value, this is the value to be used
%       If Gd = 1, random Gd value for class A
%       If Gd = 2, random Gd value for class B
%       If Gd = 3, random Gd value for class C
%       If Gd = 4, random Gd value for class E
%       If Gd = 5, random Gd value for class D
% ---- Output ----
% prof_x = X values of the profile
% prof_y = A matrix with a row for each path
%       r1(1,:) = First profile on y axis
%       r1(2,:) = Second profile on y axis
%       r1(3,:) = ....
% Gd = Roughness coeficient used
% -------------------------------------------------------------------------

% Program to generate road profiles based on Spectra implied by
% ISO standard method of reporting profile data.  ISO standard assumes
% PSD is of the form Gd(n0)*(n/n0)^-w where n0 = 0.1 cycles/m.  This method
% assumes that the pavement is isotropic.

% -------------------------------------------------------------------------

% Default values
dy = 0.1;           % Sampling frequency on Y direction

nx = L/dx + 1;      % Number of points to be calculated on X direction
nx=round(nx);

prof_x = (0:dx:L);  % X values of the profile

w = 2;              % The unevenness coefficient

%ny = myfix(Ly/dy+1,2);

ny = norm(Ly/dy+1,2);

L_diag = sqrt(((nx-1)*dx)^2+(((ny-1)*dy)^2));

dr = L_diag/(nx-1);
ISO_n = (1:nx)/L_diag;

% --- Classification of road roughness proposed by ISo ---
% A = Very Good ; Gd < 8e-6
% B = Good ; 8e-6 <= Gd < 32e-6
% C = Average ; 32e-6 <= Gd < 128e-6
% D = Poor ; 128e-6 <= Gd < 512e-6
% E = Very Poor ; 512e-6 <= Gd < 2048e-6
if Gd == 1    % ISO Class A
    Gd=4e-6+4e-6*rand;
elseif Gd == 2 % ISO Class B
    Gd=8e-6+24e-6*rand;
elseif Gd == 3 % ISO Class C
    Gd=32e-6+96e-6*rand;
elseif Gd == 4 % ISO Class D
    Gd=128e-6+384e-6*rand;
elseif Gd == 5 % ISO Class E
    Gd=512e-6+1536e-6*rand;
end 

% ???????????????????? UNKNOWN ????????????????????????????
% % ---- ISO spectrum -----
% As defined in the standard:
% ISO_spectrum = Gd*(0.1/ISO_n).^(w)*dr; ISO_spectrum(1) = 0.0;

% Modified version of form in Newland 
% Has same values as the ISO form at n0 (0.1) but does not go to infinity at n=0.
% Increasing alpha reduces differences between this form and the ISO equation.  

alpha=1000;

ISO_spectrum=((alpha+1)*Gd./(1+alpha*(ISO_n./0.1).^w))/dr/2;

% % No idea why it this must be done
Spectrum = ISO_spectrum.*linspace(1,0,length(ISO_spectrum));
Spectrum = Spectrum + Spectrum(end:-1:1);

auto_corr=real(ifft(Spectrum)); %*N_1D*dx;

L1=(1:nx-1)*dx/dr;
F1=L1-floor(L1);
L2=(nx-1:-1:1)*dx/dr;
F2=L2-floor(L2);
auto_corr_carpet= ((nx-1:-1:1).*(auto_corr(floor(L1)+1).*(1-F1)+auto_corr(ceil(L1)+1).*F1)...
      + (1:nx-1).*(auto_corr(floor(L2)+1).*(1-F2)+auto_corr(ceil(L2)+1).*F2))./nx;   
auto_corr_carpet = [auto_corr(1),auto_corr_carpet]';   
 
for k=1:ny-1
      L1=k*dy/dr;
      F1=L1-floor(L1);
      L3=(ny-k)*dy/dr;
      F3=L3-floor(L3);
      auto_corr_carpet(1,k+1)= ((ny-k)*(auto_corr(floor(L1)+1)*(1-F1)+auto_corr(ceil(L1)+1)*F1)...
      +k*(auto_corr(floor(L3)+1)*(1-F3)+auto_corr(ceil(L3)+1)*F3))./ny;   
end

for k=1:ny-1
    L1=round(sqrt(((1:nx-1)*dx).^2+(k*dy)^2)/dr);
    L2=round(sqrt(((nx-(1:nx-1))*dx).^2+(k*dy)^2)/dr);
    L3=round(sqrt(((1:nx-1)*dx).^2+((ny-k)*dy)^2)/dr);
    L4=round(sqrt(((nx-(1:nx-1))*dx).^2+((ny-k)*dy)^2)/dr);
    F1=L1-floor(L1);
    F2=L2-floor(L2);
    F3=L3-floor(L3);
    F4=L4-floor(L4);

    auto_corr_carpet(2:end,k+1)= ((nx-(1:nx-1))*(ny-k).*(auto_corr(floor(L1)+1).*(1-F1) + ...
        auto_corr(ceil(L1)+1).*F1) + ...
        (1:nx-1)*(ny-k).*(auto_corr(floor(L2)+1).*(1-F2) + auto_corr(ceil(L2)+1).*F2) + ...
        (nx-(1:nx-1))*k.*(auto_corr(floor(L3)+1).*(1-F3) + auto_corr(ceil(L3)+1).*F3) + ...
        (1:nx-1)*k.*(auto_corr(floor(L4)+1).*(1-F4) + auto_corr(ceil(L4)+1).*F4))./(ny*nx);   
end

phase=2*pi*rand(nx,ny);

phase(1,1)=0.0;
phase((nx-1)/2+1,1)=0;
if mod(nx/2,round(nx/2)) == 0
phase(nx/2+1,1)=0;    
phase(nx/2+1,(ny-1)/2+1)=0;
else
phase(1,(ny-1)/2+1)=0;
phase((nx-1)/2+1,(ny-1)/2+1)=0;
end
for j=0:(nx-1)/2-1
   phase(nx-j,1)=-phase(j+2,1);
   phase(nx-j,(ny-1)/2+1)=-phase(j+2,(ny-1)/2+1);
end
for k=0:(ny-1)/2-1
   phase(1,ny-k)=-phase(1,k+2);
end
for j=0:nx-2
   for k=0:(ny-1)/2-2
      phase(nx-j,ny-k)=-phase(j+2,k+2);
   end
end
spectrum_carpet=sqrt(abs(fft2(auto_corr_carpet))./(nx*ny));

carpet=(real(ifft2(spectrum_carpet.*exp(i*phase))*nx*ny))';

% ???????????????????? UNKNOWN ????????????????????????????

% ------------ Generation of Outputs -------------
% Selection of paths for Output
aux1 = (0:dy:Ly);
prof_y = zeros(length(paths),nx);
for k = 1:length(paths)
    aux2 = find(aux1==paths(k));
    if isempty(aux2)
        aux2 = find(aux1>paths(k),1,'first');
        aux3 = find(aux1<paths(k),1,'last');
        prof_y(k,:) = interp2(prof_x,[aux1(aux2),aux1(aux3)],[carpet(aux2,:);carpet(aux3,:)],prof_x,paths(k));
    else
        prof_y(k,:) = carpet(aux2,:);       % Output
    end
end

% % ------------ Plotting ------------
% % ---- Carpet ----
figure; pcolor((0:dx:L),(0:dy:Ly),carpet); shading interp
    xlabel 'X direction (m)'; ylabel 'Y direction (m)';
% ---- Paths -----
figure; plot(prof_x,prof_y);
    xlabel 'X direction (m)'; ylabel 'Elevation (m)';
% ---- Carpet for P010 - Paper1 ----
% % Inputs
% L_prof = 30
% whb(2) = 12
% ISOtype = 2
% dx = 0.1
carpet = carpet-min(min(carpet));
figure; surf((0:size(carpet,2)-1)*dx,(0:size(carpet,1)-1)*dy,carpet); shading interp
%set(gca,'DataAspectRatio',[10 10 1e-4]); axis tight;
set(gca,'fontsize',12)
ax = gca;
ax.YGrid = 'on';
ax.XGrid = 'on';
set(gca,'TickLabelInterpreter', 'latex')
set(gca,'XTick',[0,50,100 115]);
set(gca,'YTick',[0,3,6]); 
%set(gca,'YTick',[5,10,15]); ylim([0 15]);
%set(gca,'ZTick',[0.02]); zlim([0 0.02]);
colormap gray

axis tight;
box on;
box off;
grid off;
xlabel (' Longitudinal Distance (m)','interpreter','latex'); ylabel('Lateral Distance (m)','interpreter','latex'); zlabel ('Elevation (m)','interpreter','latex');
%J_graphs(1,30,20)
    
% ---- End of script ----