function [fg] = greenDecrease1(fg,bg1,bg2,unknow,alpha,choice)
%% 沾色和合成背景放在同一个循环里
% 效果：速度和初版本持平，即绿色通道减值，沾色，合成背景，分开操作
    height = size(fg,1); width = size(fg,2);
    fgR = fg(:,:,1); % red   channel
    fgG = fg(:,:,2); % green channel
    fgB = fg(:,:,3); % blue  channel
    
    alpha3d = zeros(height,width,3);
    for i = 1:3
        alpha3d(:,:,i) = alpha;
    end
    
    for i=1:height
        for j=1:width
            if unknow(i,j)>0.01
                switch(choice)
                    case 1, 
                        if fgG(i,j) > (fgR(i,j)+fgB(i,j))/2
                            fg(i,j,2) = (fgR(i,j)+fgB(i,j))/2 ;
                            fg(i,j,:) = alpha3d(i,j,:) .* fg(i,j,:) + bg1(i,j,:) .* (1-alpha3d(i,j,:));
                            fg(i,j,:) = alpha3d(i,j,:) .* fg(i,j,:) + bg2(i,j,:) .* (1-alpha3d(i,j,:));
                        end
                    case 2,
                        if fgG(i,j) > (2*fgB(i,j)+fgR(i,j))/3
                            fg(i,j,2) = (2*fgB(i,j)+fgR(i,j))/3 ;
                        end
                    case 3,
                        if fgG(i,j) > (2*fgR(i,j)+fgB(i,j))/3
                            fg(i,j,2) = (2*fgR(i,j)+fgB(i,j))/3 ;
                        end
                     case 4,
                        if fgG(i,j) > fgB(i,j)
                            fg(i,j,2) = fgB(i,j);
                        end
                    case 5,
                        if fgG(i,j) > fgR(i,j)
                            fg(i,j,2) = fgR(i,j) ;
                        end
                    case 6,
                        if  fgG(i,j)>fgB(i,j) && fgG(i,j)>fgR(i,j)
                            fg(i,j,2)=fg(i,j,2)-0.5*abs(2*fg(i,j,2)-fg(i,j,1)-fg(i,j,3));
                        end          
                end
            else
                fg(i,j,:) = alpha3d(i,j,:) .* fg(i,j,:) + bg2(i,j,:) .* (1-alpha3d(i,j,:));
            end
        end
    end

end