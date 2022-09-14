function [fg] = greenDecrease(fg,unknow,choice)
% @author:Tanlang
% @illustration:
    % there are five tactics to process color spill

    fgR = fg(:,:,1); % red   channel
    fgG = fg(:,:,2); % green channel
    fgB = fg(:,:,3); % blue  channel

    height = size(fg,1); width = size(fg,2);
    for i=1:height
        for j=1:width
            if unknow(i,j)>0.01
                switch(choice)
                    case 1, 
                        if fgG(i,j) > (fgR(i,j)+fgB(i,j))/2
                            fg(i,j,2) = (fgR(i,j)+fgB(i,j))/2 ;
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
                continue;
            end
        end
    end

end