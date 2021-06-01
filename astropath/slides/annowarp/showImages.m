function showImages(O)
    n=2;
    m=1;
    k = 1;
    figure(k);
        imshow(imresize(O.b,0.125));
        axis equal
        title('AP');
        box on;
        k = k+1;
    figure(k);
        imshow(2*imresize(O.q,0.125));
        axis equal
        title('QPTiff');
        box on
        k = k+1;
    shg;
end