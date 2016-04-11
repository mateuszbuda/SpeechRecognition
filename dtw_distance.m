function dtw = dtw_distance(A, B)
    % This function calculates the dynamic time warping distance among two matrices.
    % For Dynamic Time Warping (DTW), please, see E. Keogh, C. A. Ratanamahatana, Exact Indexing of Dynamic Time Warping

    sizeA = size(A,2);
    sizeB = size(B,2);
    DTW = zeros(sizeA + 1, sizeB + 1) + Inf;
    DTW(1, 1) = 0;
    
    
     % Calculate Euclidean distances between each point repeatedly
    for i=1:sizeA
        for j=1:sizeB
         edist(i,j) = norm (A(:,i)- B(:,j));   
        end
    end
    
    for m=1:sizeA
        for n=1:sizeB
            DTW (m+1,n+1) = edist(m, n) + min( [DTW(m, n+1), DTW(m+1, n), DTW(m, n)] ); 
            % Calculate the DTW value for the row
        end
    end
    
%     dtw = DTW(sizeA+1, sizeB+1); % set the final value of the matrix to the dtw length
    dtw = DTW;
    
end