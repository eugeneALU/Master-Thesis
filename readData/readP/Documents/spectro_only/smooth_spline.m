% Spline smoothing  (DeBoor's algorithm)
% Marquette University,   Milwaukee, WI  USA
% Copyright 2001 - All rights reserved.
% Fred Frigo
% 
% Dec 8, 2001
%
% Adapted to MATLAB from the following Fortran source file
%    found at http://www.psc.edu/~burkardt/src/splpak/splpak.f90

function spline_sig = smooth_spline( y, dx, npoint, smooth_factor)

p=smooth_factor;
a=[npoint:4];
v=[npoint:7];
a= 0.0;
v= 0.0;


%qty=[npoint:1];
%qu=[npoint:1];
%u=[npoint:1];

x = linspace(0.0, (npoint-1.0)/npoint , npoint);

% setupq
  v(1,4) = x(2)-x(1);
  
  for i = 2:npoint-1
    v(i,4) = x(i+1)-x(i);
    v(i,1) = dx(i-1)/v(i-1,4);
    v(i,2) = ((-1.0.*dx(i))/v(i,4)) - (dx(i)/v(i-1,4));
    v(i,3) = dx(i+1)/v(i,4);
  end 
  
   
  v(npoint,1) = 0.0;
  for i = 2:npoint-1
    v(i,5) = (v(i,1)*v(i,1)) + (v(i,2)*v(i,2)) + (v(i,3)*v(i,3));
  end
   
  for i = 3:npoint-1
    v(i-1,6) = (v(i-1,2)*v(i,1)) + (v(i-1,3)*v(i,2));
  end
   
  v(npoint-1,6) = 0.0;

  for i = 4: npoint-1
    v(i-2,7) = v(i-2,3)*v(i,1);
  end
   
  v(npoint-2,7) = 0.0;
  v(npoint-1,7) = 0.0;
%!
%!  Construct  q-transp. * y  in  qty.
%!
  prev = (y(2)-y(1))/v(1,4);
  for i= 2:npoint-1
    diff = (y(i+1)-y(i))/v(i,4);
    %qty(i) = diff-prev;
    a(i,4) = diff - prev;
    prev = diff;
  end 
  
% end setupq  

%chol1d

%!
%!  Construct 6*(1-p)*q-transp.*(d**2)*q + p*r
%!
  six1mp = 6.0.*(1.0-p);
  twop = 2.0.*p;
  
  for i = 2: npoint-1
    v(i,1) = (six1mp.*v(i,5)) + (twop.*(v(i-1,4)) + v(i,4));
    v(i,2) = (six1mp.*v(i,6)) +( p.*v(i,4));
    v(i,3) = six1mp.*v(i,7);
  end 
  
  if ( npoint < 4 ) 
    u(1) = 0.0;
    u(2) = a(2,4)/v(2,1);
    u(3) = 0.0;
%!
%!  Factorization
%!
  else
    for i = 2: npoint-2;
      ratio = v(i,2)/v(i,1);
      v(i+1,1) = v(i+1,1)-(ratio.*v(i,2));
      v(i+1,2) = v(i+1,2)-(ratio.*v(i,3));
      v(i,2) = ratio;
      ratio = v(i,3)./v(i,1);
      v(i+2,1) = v(i+2,1)-(ratio.*v(i,3));
      v(i,3) = ratio;
    end 
%!
%!  Forward substitution
%!
    a(1,3) = 0.0;
    v(1,3) = 0.0;
    a(2,3) = a(2,4);
    for i = 2: npoint-2
      a(i+1,3) = a(i+1,4) - (v(i,2)*a(i,3)) - (v(i-1,3)*a(i-1,3));
    end 
%!
%!  Back substitution.
%!
    a(npoint,3) = 0.0;
    a(npoint-1,3) = a(npoint-1,3) / v(npoint-1,1);

    for i = npoint-2:-1:2
      a(i,3) = (a(i,3)/v(i,1)) - (a(i+1,3)*v(i,2)) - (a(i+2,3)*v(i,3));
    end 

  end
%!
%!  Construct Q*U.
%!
  prev = 0.0;
  for i = 2: npoint
    a(i,1) = (a(i,3)-a(i-1,3))/v(i-1,4);
    a(i-1,1) = a(i,1)-prev;
    prev = a(i,1);
  end 

  a(npoint,1) = -1.0.*a(npoint,1);
    
%end chol1d

  for i = 1: npoint
    spline_sig(i) = y(i)-(6.0.*(1.0-p).*dx(i).*dx(i).*a(i,1));
  end
  
%  for i = 1: npoint
%    a(i,3) = 6.0*a(i,3)*p;
%  end 

%  for i = 1: npoint-1
%    a(i,4) = (a(i+1,3)-a(i,3))/v(i,4);
%    a(i,2) = (a(i+1,1)-a(i,1))/v(i,4)-(a(i,3)+a(i,4)/3.*v(i,4))/2.*v(i,4);
%  end 
 