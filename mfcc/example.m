% EXAMPLE Simple demo of the MFCC function usage.
%
%   This script is a step by step walk-through of computation of the
%   mel frequency cepstral coefficients (MFCCs) from a speech signal
%   using the MFCC routine.
%
%   See also MFCC, COMPARE.

%   Author: Kamil Wojcicki, September 2011


    % Clean-up MATLAB's environment
   % clear all; close all; clc;  

    
    % Define variables
    Tw = 10;                % analysis frame duration (ms)
    Ts = 5;                % analysis frame shift (ms)
    alpha = 0.99;           % preemphasis coefficient
    M =8;                 % number of filterbank channels 
    C =5;                 % number of cepstral coefficients
    L =24;                 % cepstral sine lifter parameter
    LF = 0.01;               % lower frequency limit (Hz)
    HF = 250;              % upper frequency limit (Hz)
    wav_file = 'sp10.wav';  % input audio filename


    % Read speech samples, sampling rate and precision from file
    %[ speech, fs ] = audioread( wav_file );
    fs=250;

    % Feature extraction (feature vectors as columns)
    [ MFCCs, FBEs, frames ] = ...
                    mfcc( y, 250, Tw, Ts, alpha, @hamming, [LF HF], M, C+1, L );

    [ MFCCs2, FBEs2, frames2 ] = ...
                    mfcc( x, 250, Tw, Ts, alpha, @hamming, [LF HF], M, C+1, L );
    % Generate data needed for plotting 
%     [ Nw, NF ] = size( frames );                % frame length and number of frames
%     time_frames = [0:NF-1]*Ts*0.001+0.5*Nw/fs;  % time vector (s) for frames 
%     time = [ 0:length(speech)-1 ]/fs;           % time vector (s) for signal samples 
%     logFBEs = 20*log10( FBEs );                 % compute log FBEs for plotting
%     logFBEs_floor = max(logFBEs(:))-50;         % get logFBE floor 50 dB below max
%     logFBEs( logFBEs<logFBEs_floor ) = logFBEs_floor; % limit logFBE dynamic range
% 
% 
%     % Generate plots
%     figure('Position', [30 30 800 600], 'PaperPositionMode', 'auto', ... 
%               'color', 'w', 'PaperOrientation', 'landscape', 'Visible', 'on' ); 
% 
%     subplot( 311 );
%     plot( time, speech, 'k' );
%     xlim( [ min(time_frames) max(time_frames) ] );
%     xlabel( 'Time (s)' ); 
%     ylabel( 'Amplitude' ); 
%     title( 'Speech waveform'); 
% 
%     subplot( 312 );
%     imagesc( time_frames, [1:M], logFBEs ); 
%     axis( 'xy' );
%     xlim( [ min(time_frames) max(time_frames) ] );
%     xlabel( 'Time (s)' ); 
%     ylabel( 'Channel index' ); 
%     title( 'Log (mel) filterbank energies'); 
% 
%     subplot( 313 );
%     imagesc( time_frames, [1:C], MFCCs(2:end,:) ); % HTK's TARGETKIND: MFCC
%     %imagesc( time_frames, [1:C+1], MFCCs );       % HTK's TARGETKIND: MFCC_0
%     axis( 'xy' );
%     xlim( [ min(time_frames) max(time_frames) ] );
%     xlabel( 'Time (s)' ); 
%     ylabel( 'Cepstrum index' );
%     title( 'Mel frequency cepstrum' );
% 
%     % Set color map to grayscale
%     colormap( 1-colormap('gray') ); 
% 
%     % Print figure to pdf and png files
%     print('-dpdf', sprintf('%s.pdf', mfilename)); 
%     print('-dpng', sprintf('%s.png', mfilename)); 


%%
  figure(256)
  for i=1:size(MFCCs,1)
      plot(MFCCs(i,:))
      hold on
      plot(MFCCs2(i,:))
  end
 