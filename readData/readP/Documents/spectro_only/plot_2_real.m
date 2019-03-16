% plot_2_real -  Plots 2 real valued vectors
% Marquette University,   Milwaukee, WI  USA
% Copyright 2004 - All rights reserved.
% Fred J. Frigo
%
% Mar 5, 2004   - Original
%                

function  plot_2_real( plot1_label, input_data1, plot2_label, input_data2 )

   % get size of vector
   vector_size = max(size(input_data1));
   x = [ 1:vector_size];

   figure;  
   subplot(2,1,1);  
   plot(x,input_data1,'k');
   ylabel(plot1_label);
   xlabel('time');
   
   if (nargin > 2)
      subplot(2,1,2);
      plot(x, input_data2, 'k');
      ylabel(plot2_label);
      xlabel('time');
   end