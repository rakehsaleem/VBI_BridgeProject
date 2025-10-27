%Removal of vehicle Response
%Written by Muhammad Zohaib Sarwar
%Dated 1 january 2020
clc
close all
clear all
%% Data Loading

Data_Label0=load('13102021_Five_Axle_profile_healthy_Fixed_Speed')
Data_Label1=load('13102021_Five_Axle_profile_damage_Fixed_Speed_50_30')
Data_Label2=load('13102021_Five_Axle_profile_damage_Fixed_Speed_50_10')
Data_Label3=load('13102021_Five_Axle_profile_damage_Fixed_Speed_375_30')
Data_Label4=load('13102021_Five_Axle_profile_damage_Fixed_Speed_375_10')
Data_Label5=load('13102021_Five_Axle_profile_damage_Fixed_Speed_25_30')
% Data_Label6=load('13102021_Five_Axle_profile_damage_Fixed_Speed_25_10')
Data_Label7=load('13102021_Five_Axle_profile_damage_Fixed_Speed_125_30')
% Data_Label8=load('13102021_Five_Axle_profile_damage_Fixed_Speed_125_10')


%% muhammas zOHA A FSrwar
%%
Fs=1000;
R=500;
T=1000;
n=3;
level=8


tic
for k=1:1:T

Vel(k)= Data_Label0.MCSol.Run(k).Veh.Pos.vel;

S01=             Data_Label0.MCSol.Run(k).Sol.Veh.A(5,:);
S02=             Data_Label0.MCSol.Run(k).Sol.Veh.A(7,:);
% S03=             Data_Label0.MCSol.Run(k).Sol.Veh.A(8,:); 
% S04=             Data_Label0.MCSol.Run(k).Sol.Veh.A(7,:);
S05=             Data_Label0.MCSol.Run(k).Sol.Veh.A(8,:);  


S0{:,k}=[wav_cust(S01(Data_Label0.MCSol.Run(k).Veh.Pos.t0_ind_beam:end),n,level);...
         wav_cust(S02(Data_Label0.MCSol.Run(k).Veh.Pos.t0_ind_beam:end),n,level);...
         wav_cust(S05(Data_Label0.MCSol.Run(k).Veh.Pos.t0_ind_beam:end),n,level)];
   
end

for k=1:1:R

S11=             Data_Label1.MCSol.Run(k).Sol.Veh.A(5,:);
S12=             Data_Label1.MCSol.Run(k).Sol.Veh.A(7,:);
% S13=             Data_Label1.MCSol.Run(k).Sol.Veh.A(6,:); 
% S14=             Data_Label1.MCSol.Run(k).Sol.Veh.A(7,:);
S15=             Data_Label1.MCSol.Run(k).Sol.Veh.A(8,:);  

S1{:,k}=[wav_cust(S11(Data_Label1.MCSol.Run(k).Veh.Pos.t0_ind_beam:end),n,level);...
         wav_cust(S12(Data_Label1.MCSol.Run(k).Veh.Pos.t0_ind_beam:end),n,level);...
         wav_cust(S15(Data_Label1.MCSol.Run(k).Veh.Pos.t0_ind_beam:end),n,level)];


S21=             Data_Label2.MCSol.Run(k).Sol.Veh.A(5,:);
S22=             Data_Label2.MCSol.Run(k).Sol.Veh.A(7,:);
% S23=             Data_Label2.MCSol.Run(k).Sol.Veh.A(6,:);
% S24=             Data_Label2.MCSol.Run(k).Sol.Veh.A(7,:);
S25=             Data_Label2.MCSol.Run(k).Sol.Veh.A(8,:);  

S2{:,k}=[wav_cust(S21(Data_Label2.MCSol.Run(k).Veh.Pos.t0_ind_beam:end),n,level);...
         wav_cust(S22(Data_Label2.MCSol.Run(k).Veh.Pos.t0_ind_beam:end),n,level);...
         wav_cust(S25(Data_Label2.MCSol.Run(k).Veh.Pos.t0_ind_beam:end),n,level)];


S31=             Data_Label3.MCSol.Run(k).Sol.Veh.A(5,:);
S32=             Data_Label3.MCSol.Run(k).Sol.Veh.A(7,:);
S35=             Data_Label3.MCSol.Run(k).Sol.Veh.A(8,:);  

S3{:,k}=[wav_cust(S31(Data_Label3.MCSol.Run(k).Veh.Pos.t0_ind_beam:end),n,level);...
         wav_cust(S32(Data_Label3.MCSol.Run(k).Veh.Pos.t0_ind_beam:end),n,level);...
         wav_cust(S35(Data_Label3.MCSol.Run(k).Veh.Pos.t0_ind_beam:end),n,level)];



S41=             Data_Label4.MCSol.Run(k).Sol.Veh.A(5,:);
S42=             Data_Label4.MCSol.Run(k).Sol.Veh.A(7,:);
S45=             Data_Label4.MCSol.Run(k).Sol.Veh.A(8,:);  

S4{:,k}=[wav_cust(S41(Data_Label4.MCSol.Run(k).Veh.Pos.t0_ind_beam:end),n,level);...
         wav_cust(S42(Data_Label4.MCSol.Run(k).Veh.Pos.t0_ind_beam:end),n,level);...
         wav_cust(S45(Data_Label4.MCSol.Run(k).Veh.Pos.t0_ind_beam:end),n,level)];


S51=             Data_Label5.MCSol.Run(k).Sol.Veh.A(5,:);
S52=             Data_Label5.MCSol.Run(k).Sol.Veh.A(7,:);
S55=             Data_Label5.MCSol.Run(k).Sol.Veh.A(8,:);  

S5{:,k}=[ wav_cust(S51(Data_Label5.MCSol.Run(k).Veh.Pos.t0_ind_beam:end),n,level);... 
          wav_cust(S52(Data_Label5.MCSol.Run(k).Veh.Pos.t0_ind_beam:end),n,level);...
          wav_cust(S55(Data_Label5.MCSol.Run(k).Veh.Pos.t0_ind_beam:end),n,level)];

% 
% S61=             Data_Label6.MCSol.Run(k).Sol.Veh.A(4,:);
% S62=             Data_Label6.MCSol.Run(k).Sol.Veh.A(5,:);
% S63=             Data_Label6.MCSol.Run(k).Sol.Veh.A(6,:);
% S64=             Data_Label6.MCSol.Run(k).Sol.Veh.A(7,:);
% S65=             Data_Label6.MCSol.Run(k).Sol.Veh.A(8,:);  

% S6{:,k}=[  wav_cust(S61(Data_Label6.MCSol.Run(k).Veh.Pos.t0_ind_beam:end),n,level);... 
%            wav_cust(S62(Data_Label6.MCSol.Run(k).Veh.Pos.t0_ind_beam:end),n,level);...
%            wav_cust(S63(Data_Label6.MCSol.Run(k).Veh.Pos.t0_ind_beam:end),n,level);... 
%            wav_cust(S64(Data_Label6.MCSol.Run(k).Veh.Pos.t0_ind_beam:end),n,level);...
%            wav_cust(S65(Data_Label6.MCSol.Run(k).Veh.Pos.t0_ind_beam:end),n,level)];



S71=             Data_Label7.MCSol.Run(k).Sol.Veh.A(5,:);
S72=             Data_Label7.MCSol.Run(k).Sol.Veh.A(7,:);
S75=             Data_Label7.MCSol.Run(k).Sol.Veh.A(8,:);

S7{:,k}=[wav_cust(S71(Data_Label7.MCSol.Run(k).Veh.Pos.t0_ind_beam:end),n,level);...
         wav_cust(S72(Data_Label7.MCSol.Run(k).Veh.Pos.t0_ind_beam:end),n,level);...
         wav_cust(S75(Data_Label7.MCSol.Run(k).Veh.Pos.t0_ind_beam:end),n,level)];


% S81=             Data_Label8.MCSol.Run(k).Sol.Veh.A(4,:);
% S82=             Data_Label8.MCSol.Run(k).Sol.Veh.A(5,:);
% S83=             Data_Label8.MCSol.Run(k).Sol.Veh.A(6,:);
% S84=             Data_Label8.MCSol.Run(k).Sol.Veh.A(7,:);
% S85=             Data_Label8.MCSol.Run(k).Sol.Veh.A(8,:);  
% 
% S8{:,k}=[ wav_cust(S81(Data_Label8.MCSol.Run(k).Veh.Pos.t0_ind_beam:end),n,level);... 
%           wav_cust(S82(Data_Label8.MCSol.Run(k).Veh.Pos.t0_ind_beam:end),n,level);...
%           wav_cust(S83(Data_Label8.MCSol.Run(k).Veh.Pos.t0_ind_beam:end),n,level);... 
%           wav_cust(S84(Data_Label8.MCSol.Run(k).Veh.Pos.t0_ind_beam:end),n,level);... 
%           wav_cust(S85(Data_Label8.MCSol.Run(k).Veh.Pos.t0_ind_beam:end),n,level)];
end




%%
new_X=[ S0,S1,S1,S2,S2,S3,S3,S4,S4,S5,S5,S7,S7];
new_labels=[1.* ones(1,T),2.* ones(1,R*2),3.* ones(1,R*2),4.* ones(1,R*2),5.* ones(1,R*2),6.* ones(1,R*2),7.* ones(1,R*2)];

% new_X=[ S0,S1,S2,S3,S4,S5,S6,S7,S8];
% new_labels=[1.* ones(1,T),2.* ones(1,R),3.* ones(1,R),4.* ones(1,R),5.* ones(1,R),6.* ones(1,R),7.* ones(1,R),8.* ones(1,R),9.* ones(1,R)];


save('D:\Zohaib_Phd Folder\Paper 2\Matlab Model\GVW_Paper\Python Folder\MLSTM-FCN-master\data\arabic_voice\arabic_voice_window_3_ifperm_vehicle_5_axle_fixed_speed_wavlet_Low_sensor.mat','new_X','new_labels')
