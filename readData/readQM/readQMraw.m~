function [RealQrapImage TIvec TEvec] = readQMraw(par)


[Xsize Ysize Zsize] = size(par.image(:,:,find(par.echo_number==1 & ...
            par.dynamic_scan_number==1 & ...
            par.image_type_mr == 0)));
        
Zero1vol = zeros(Xsize, Ysize, Zsize);
        
phaseImage = zeros(Xsize, Ysize, Zsize, par.Max_number_of_echoes);

phase=(par.Max_number_of_dynamics-1):-1:0;
repp_shift = (par.Max_number_of_slices-par.Max_number_of_dynamics-1)/((par.Max_number_of_dynamics-1)^2);
time_step = par.TR/par.Max_number_of_slices;       
        % phas qrapmaster images
% returns RealQrapImage{echo,dynamic}
VolIDX = 0;



for echo=1:par.Max_number_of_echoes  
    
    % Create empty image volume for summing all dynamics for each echo
    sumQrapEcho=Zero1vol;
        
        for dynamic=1:par.Max_number_of_dynamics
        
            complexIm=par.image(:,:,find(par.echo_number==echo & ...
                par.dynamic_scan_number==dynamic & ...
                par.image_type_mr == 0)) .* exp(i *...
                par.image(:,:,find(par.echo_number==echo & ...
                par.dynamic_scan_number==dynamic & ...
                par.image_type_mr == 3)));
         
            sumQrapEcho=complexIm+sumQrapEcho;
        
        end


        phaseImage(:,:,:,echo)=angle(sumQrapEcho);
        
        for dynamic=1:par.Max_number_of_dynamics
        
            VolIDX = VolIDX + 1;
        
        % Calc qrap TI
        TIvec(VolIDX)= time_step*(par.Max_number_of_dynamics-phase(end-dynamic+1)+ floor(repp_shift*((par.Max_number_of_dynamics-phase(end-dynamic+1)-1)^2)));
        TEvec(VolIDX) = unique(par.echo_time(find(par.echo_number==echo)));
        % Get phased image for each echo at each phase
        
        RealQrapImage(:,:,:,VolIDX) = real(par.image(:,:,find(par.echo_number==echo & ...
            par.dynamic_scan_number==dynamic & ...
            par.image_type_mr == 0)) .* exp(i *...
            par.image(:,:,find(par.echo_number==echo & ...
            par.dynamic_scan_number==dynamic & ...
            par.image_type_mr == 3))) .*exp(-i * phaseImage(:,:,:,echo)));
        
        end
      
end
