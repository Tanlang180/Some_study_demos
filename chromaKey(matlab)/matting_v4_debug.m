close all;
clear all;
clc;

%% set fg|bg path and save path
imgPath =  '.\images\foreground\'; 
allImgs = dir([imgPath,'*.jpg']);
background = 'images/background/bg2.jpg';

downSample_ratio = 1;  % 下采样率，过大会导致图像不清晰
%% knn kernel filter matting
disp('Chromakeing')
for i = 4:length(allImgs)
    %% read bg and fg
    fg = double(imread([imgPath, allImgs(i).name]));
    height = size(fg,1); width = size(fg,2);
    fg = double(imresize(fg, [floor(height/downSample_ratio),floor(width/downSample_ratio)], 'bilinear'));
    
    bgColor = mean(mean(imcrop(fg,[0,0,50,50])));   
    
    bg1 = double(180.*ones(floor(height/downSample_ratio),floor(width/downSample_ratio),3));
    bg2 = double(255.*ones(floor(height/downSample_ratio),floor(width/downSample_ratio),3));
    
    t1 = clock();
    %% get the related matrix 
    [alpha,unknow,fgRegion] = genMaskMatrix(fg,2);
    
    %% 公式法计算前景，无效
    img = blengding(fg,alpha,bgColor,bg1);
    
    %% six methods to decrease green channel value     
%     img_comp = greenDecrease1(fg,bg1,bg2,unknow,alpha,1);
    img_comp = greenDecrease2(fg,bg1,bg2,unknow,alpha,1);
    %% kernel filter from principle of locality 
%     img_filter = KnnKernelFilter1(img_comp,unknow,alpha,fgRegion,9,10,15);
     img_filter = KnnKernelFilter2(img_comp,unknow,fgRegion,7);
%      img_filter = KnnKernelFilter3(img_comp,unknow,fgRegion,5);
    %% image_edge process   
%     img_result = Gaussion_edge1(img_comp,unknow,5,1,3);
    img_result = Gaussion_edge3(img_filter,unknow,3,1);
    img_result = imresize(img_result,[height,width], 'bilinear');
    
    %% 图像处理帧率显示
    process = sprintf('Process has finished [ %d / %d]',i,length(allImgs));
    disp(process);
    disp(['帧率 ： ',num2str(1/etime(clock,t1)),'帧']);
    %% show image
    figure(2)
    imshow(img_comp./255);
%     figure(3)
%     imshow(img_filter./255);
%     figure(4)
%     imshow(unknow)
    figure(5)
    imshow(img_result./255);

end
disp('All jobs has finished!');
