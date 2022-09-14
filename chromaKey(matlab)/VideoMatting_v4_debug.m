% @author:Tanlang
% @illustration: chroma key project

close all;
clear all;
clc;

%% 3个实例视频
% vidioPath = 'images\JiGong.mp4';        % 640*368
% vidioPath = 'G:\Dataset\Videos\woman2.mp4';    % 3840*2160
vidioPath = 'G:\Dataset\Videos\human.mp4';    % 720x1280

%% Set params
obj = VideoReader(vidioPath);
numFrames = obj.NumberOfFrames;
width = obj.Width;
height = obj.Height;
downSample_ratio = 1;  % 下采样率，过大会导致图像不清晰

bg1 = double(180.*ones(floor(height/downSample_ratio),floor(width/downSample_ratio),3));
background = 'images/background/bg2.jpg';
bg2 = double(zeros(floor(height/downSample_ratio),floor(width/downSample_ratio),3));
% bg2 = imread(background);
% bg2 = double(imresize(bg2, [floor(height/downSample_ratio),floor(width/downSample_ratio)], 'bilinear'));

%% process video
disp('Chromakeing')
for i = 1:numFrames
    frame = read(obj,i);

    %% Calculating matrices
    fg = double(imresize(frame, [floor(height/downSample_ratio),floor(width/downSample_ratio)], 'bilinear'));
    t1 = clock();
   %% get the related matrix
    [alpha,unknow,fgRegion] = genMaskMatrix(fg,1);
    
    %% six methods to decrease green channel value and composite in new background
    % type 1 ,速度 rank 2
%     imgGd = greenDecrease(fg,unknow,2);
%     img_comp = blendingImage(imgGd,bg1,alpha);  % 沾色
%     img_comp = blendingImage(img_comp,bg2,alpha);
    %type 2 ,速度 rank 3
%     img_comp = greenDecrease1(fg,bg1,bg2,unknow,alpha,1);
    %type 3 ,速度 rank 1
    img_comp = greenDecrease2(fg,bg1,bg2,unknow,alpha,1);

    %% kernel filter from principle of locality 
%     img_filter = KnnKernelFilter1(img_comp,unknow,alpha,fgRegion,9,10,15);
    img_filter = KnnKernelFilter2(img_comp,unknow,fgRegion,5);
%     img_filter = KnnKernelFilter3(img_comp,unknow,fgRegion,5);
    %% image_edge process   
%     img_result = Gaussion_edge1(img_comp,unknow,5,1,3);
%     img_result = Gaussion_edge2(img_comp,unknow,5,1,3);
    img_result = Gaussion_edge3(img_filter,unknow,5,1);
%     img_result = Gaussion_edge4(img_comp,unknow,5,1);

    img_result = imresize(img_result,[height,width], 'bilinear');

    %% 显示帧率
    process = sprintf('Process has finished [ %d / %d]',i,numFrames);
    disp(process);
    disp(['帧率 ： ',num2str(1/etime(clock,t1)),'帧']);
    
%     figure(1)
%     imshow(img./255);
%     figure(2)
%     imshow(img_comp./255);
%     figure(3)
%     imshow(alpha)
%     figure(4)
%     imshow(unknow)
%     figure(5)
%     imshow(img_filter./255)
%     figure(6)
%     imshow(img_result./255);
%         
end
disp('All work has been finished!!!');

