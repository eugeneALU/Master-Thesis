% plot_complex -  Plots real, imag and magnitude values of complex vector
% Marquette University,   Milwaukee, WI  USA
% Copyright 2004 - All rights reserved.
% Fred J. Frigo
%
% Mar 5, 2004   - Original
%                
%

function  plot_complex( plot_title, input_data )

   % get size of vector
   vector_size = max(size(input_data));
   x = [ 1:vector_size];

   figure;  
   subplot(3,1,1);  
   plot(x,real(input_data),'k');
   title(plot_title);
   ylabel('Real');
   xlabel('time');
   
   subplot(3,1,2);
   plot(x, imag(input_data), 'k');
   ylabel('Imaginary');
   xlabel('time');
 
   subplot(3,1,3);
   plot(x, abs(input_data), 'k');
   ylabel('Magnitude');
   xlabel('time');
   
   % Entire phase correction vector has magnitude of 1.0
   % Check for entire vector with magnitude = 1.0
   if(( max(abs(input_data) < 1.0000001 )) && ...
      ( min(abs(input_data) > 0.999999  )))
      my_axis=axis;
      my_axis(3)=0.0;  % ymin
      my_axis(4)=1.5;  % ymax
      axis(my_axis);
   end
