clear;
close all;
clc;

%% BCS reconstruction of prospectively undersampled data

pathToData = '/home/kostya/Documents/BCS_matlab_code/BCS_pros/';
folder = [44];

if(~exist('NIIfolder', 'var'))
    NIIfolder = [pathToData, 'NIFTI'];
end
    
if(~exist(NIIfolder,'dir'))
    mkdir([pathToData ,NIIfolder]);
end

PFfactor = 1.0;
Rfactor = 2; % reduction factor
ImageCollector = zeros(48,96,5);

for index=1:length(folder)
    pathName = [pathToData, sprintf('%d',folder(index))]; tic

    [kSpace,kMask] = readFID(pathName);
    [RO_steps,PE_steps,NR,NSlices] = size(kSpace);
    
    pointPerSlice = RO_steps*PE_steps;
    X_res = [];
    % Slice loop
    for slice=1:NSlices
        mask_tmp = squeeze(kMask(:,:,:,slice));
        k_tmp = squeeze(kSpace(:,:,:,slice));
     
        %S=find(mask_tmp~=0);
        

        A = @(z)SampFour2D(z,mask_tmp,RO_steps,PE_steps,NR);
        At = @(z)InvSampFour2D(z,mask_tmp,RO_steps,PE_steps,NR);

        %% first guess
        b = reshape(fftshift(fftshift(k_tmp,1),2),pointPerSlice,NR);
        
        %x_init = At(b);
        %x_init1=reshape(x_init,RO_steps,PE_steps,NR);

        %% The algorithm parameters are all specified in opts.
        opts.outer =100;% The iterations of the outer loop
        opts.inner =100;% The iterations of the inner loop
        opts.lambda1=0.0020/(RO_steps*PE_steps);%0.0020/(n1*n2);% The regularization parameter (on the l1 norm of U) 
        opts.lambda2=1e-3/(RO_steps*PE_steps);%1e-3/(n1*n2);% The regularization parameter (on the l2 norm of V)
        opts.lambda3=0.001;%0.01; % The regularization parameter for low rank constraint
        opts.beta1=0.5e-3;%1e-3; % continuation parameter for the l1 norm; initialize it

        %% r denotes the number of temporal basis functions in the dictionary
        NumTempBasisFunc = 50;
        for num = 1:length(NumTempBasisFunc)

        %% Initialize the U and V matrices; (spatial weights/coefficients and temporal bases)
        % V - random matrix; U from the CG subproblem based on random V
        earray=zeros(1, 50);rmse_un=[];earrayout=[];
        V = double(rand(NumTempBasisFunc(num),NR)); 
        V_init = V;

        %  CG(conjugate gradient) algorithm to solve the U initiaization subproblem:

        [U,earray_u] = xupdateUal(b,A, At,zeros(pointPerSlice,NumTempBasisFunc(num),'double'), V,0,opts,1e-10,30);

        iii=1;succ_error=[]; X_est=U*V; X_est = reshape(X_est,RO_steps,PE_steps,NR);opts.beta1=1./max(abs(U(:))); 
        S = zeros(RO_steps*PE_steps,NR);
        for out = single(1: opts.outer)
           for in = single(1:opts.inner)
               %% L subproblem; L - auxillary variable (Shrinkage of spatial weights)

               X_iter=X_est;
               Z = U;
               L_1 = (abs(Z)-1/opts.beta1); 
               L_1 = L_1.*(L_1>0); 
               L_2 = abs(Z)+(abs(Z)<1e-12);
               L = L_1.*Z./L_2;

               %% U subproblem - CG algorithm

               [U,earray_u] = xupdateUal(b,A, At,zeros(size(U)), V,L,opts,1e-8,30);

               %% V subproblem - CG algorithm
               [V,earray_v] = xupdateVprob(b,A, At,zeros(size(V)), U,opts,1e-8,30);

               S = U*V;

               %% COST Calculations
               dc = A(S)-b; % data consistency
               reg_U = sum(abs(U(:))); % l1 norm on the spatial weights
               reg_V = sum(abs(V(:)).^2); % l2 norm on the dictionary
               cost = sum(abs(dc(:)).^2) +opts.lambda1*reg_U+opts.lambda2*reg_V;%+opts.lambda3*reg_L;
               earray = [earray,cost];
               X_est = reshape(S,RO_steps,PE_steps,NR);

%                figure(7);
%                subplot(1,2,1); imagesc(abs(squeeze(fftshift(X_est(:,:,1),1)))'); axis image off;
%                subplot(1,2,2); plot(double(earray)); title('cost');
               pause(0.01);
               iii=iii+1;
               if in>1
                if (abs(earray(end)-earray(end-1))/abs(earray(end))) < 1e-3
                    break; % Break the inner loop if the cost stops decreasing
                end

               end
           end
           nuc_norm  = X_est; % At low \beta_{1}, the solution is an equivalent min. nuclear norm (low rank) solution
           opts.beta1 = opts.beta1*40; % update the continuation parameter
           earrayout=[earrayout, earray(end)];
           if  out>1
               abs(earrayout(end)-earrayout(end-1))/abs(earrayout(end));
           if (abs(earrayout(end)-earrayout(end-1))/abs(earrayout(end))) < 1e-8
                   break; % Break the outer loop if the cost stops decreasing
           end
           end
        end

        end
        %err = (abs(earrayout(end)-earrayout(end-1))/abs(earrayout(end)));
        
        X_est = permute(X_est,[2 1 3]);
        X_res(:,:,:,slice) = X_est;
        
    end
    save(['BCS_Flash_R',num2str(Rfactor),'_',num2str(folder(index)),'.mat'],'X_res');
    niiname = [NIIfolder,'/',num2str(folder(index)),'.nii'];
    visupars = readvisu_pars(num2str(folder(index)));
    if(visupars.dim < 3)
        NII = make_nii(abs(X_res), [visupars.extent(1)/visupars.size(1), visupars.extent(2)/visupars.size(2),visupars.thickness]);
    else
        NII = make_nii(abs(X_res), [visupars.extent(1)/visupars.size(1), visupars.extent(2)/visupars.size(2),visupars.extent(3)/visupars.size(3)]);
    end
    save_nii(NII,niiname);
    toc
    %-------IMAGING PART-----------
    image_shifted = fftshift(X_est(:,:,1,1), 2);
    ImageCollector(:,:,index) = image_shifted;%[ImagCollector, image_shifted];%cat(1, ImageCollector, image_shifted);
    figure(index); imshow(abs(squeeze(image_shifted)), []); axis image off; title(['Reconstructed image #', sprintf('%d',index)]);
end

trigger = readTrigger(num2str(folder(1)));
time = linspace(0,size(trigger,1),size(trigger,1));
%plot(squeeze(time), squeeze(trigger));
fid = fopen([NIIfolder,'/trigger_delay0.txt'],'w');
for index = 1:length(trigger)
    fprintf(fid,'%d\n',trigger(index));
end
fclose(fid);