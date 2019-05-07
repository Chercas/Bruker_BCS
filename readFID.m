function [kSpace, kMask] = readFID(pathname)

%---Reading auxillary files
method = readmethod(pathname);
acqp = readacqp(pathname);
%---Number of Repetitions
NR = acqp.NR;
%---Number of Slices
NSlices = acqp.NSlices;
%---Number of Phase & RO steps
PE_steps = acqp.size(2);
RO_steps = method.kSize(1);
%---Reduction Factor
RedFactor = method.Rfactor;
%---Total PE lines
PE_total = PE_steps*RedFactor;

%---Reading FID word_by_word
FID=fopen([pathname,'/fid']);
if(strcmp(acqp.wordsize,'_32_BIT'));wSize = 'int32';else wSize = 'int16';end
RawData = fread(FID,inf,wSize,0,'l');
fclose(FID);
                                                                                                        
%---Creating complex data from words-----
RawDataComplex = RawData(1:2:end) + sqrt(-1)*RawData(2:2:end);
[TotalPoints, ~] = size(RawDataComplex); 

%---Computing an ADC sampling size (num of points/words sampled at ones)
%---should be >= then RO and be an integer value within ADC sampling range.
ADC = (0:1:31); bit = 1;
while RO_steps > power(2, ADC(bit)); bit=bit+1; end
ADC_SampPoints = power(2, ADC(bit));

%---Computing k-space lines opposite permute procedure by rel pos in 'Gradp'
index = method.Gradp2;
kLineIndex =round(PE_total*(index+1)/2 + 1);
kLineIndex=reshape(kLineIndex,[NSlices,PE_steps,NR]);

if TotalPoints == ADC_SampPoints*NSlices*PE_steps*NR
    ADCMatrix = reshape(RawDataComplex,ADC_SampPoints,NSlices,PE_steps,NR);
    RawMatrix = ADCMatrix(1:RO_steps,:,:,:);
    kSpace = zeros(size(RawMatrix));
    for i=1:NSlices
        for j=1:NR
            kSpace(:,i,kLineIndex(i,:,j),j) = RawMatrix(:,i,:,j);
        end
    end 
%---Creating Masks
kMask = cast(abs(kSpace)>0, 'double');
kMask = permute(kMask, [1 3 4 2]);
kMask = fftshift(fftshift(kMask,1),2);
kSpace = permute(kSpace, [1 3 4 2]);
end