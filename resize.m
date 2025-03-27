%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Image Segmentation and Object Detection in MATLAB
% Using a Single Figure for Visualization (Including Birds Turned Red)
%
% Demonstrates:
%   1) Preprocessing (resizing to 512x512)
%   2) Color Space Conversion (RGB, HSV, YCbCr)
%   3) Color Segmentation (Thresholding + bwareaopen)
%   4) Edge Detection (Sobel) + Edge Enhancement (Dilation)
%   5) K-means Clustering Segmentation (with noise removal)
%   6) Object Detection (Connected Components)
%   7) Overlaying a Different Color on the Objects of Interest (Turning Birds Red)
%   8) Single-Figure Visualization (10 subplots)
%
% NOTE:
%   - Threshold values here are illustrative and may need tuning.
%   - The image "birdies.jpg" should be in the current folder or path.
%   - All steps are combined into one script (no separate functions).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

close all; clear; clc;

%% 1) READ & RESIZE IMAGE
originalImg = imread('birdies.jpg');
birdImg = imresize(originalImg, [512 512]);
grayBird = rgb2gray(birdImg); % for edge detection

%% 2) COLOR SPACE CONVERSIONS
hsvBird   = rgb2hsv(birdImg);
ycbcrBird = rgb2ycbcr(birdImg);

%% 3) COLOR SEGMENTATION (RGB Example)
% Simple threshold for "dark" pixels to capture silhouettes (birds)
R = birdImg(:,:,1);
G = birdImg(:,:,2);
B = birdImg(:,:,3);

% Adjust as needed (0-255 range for uint8 images)
darkThreshold = 70; 
BW_rgb = (R < darkThreshold) & (G < darkThreshold) & (B < darkThreshold);

% Remove small objects
BW_rgb = bwareaopen(BW_rgb, 50);

% Create an overlay image showing only segmented regions
colorSegmentedOverlay = birdImg;
colorSegmentedOverlay(repmat(~BW_rgb, [1 1 3])) = 0;

%% 4) EDGE DETECTION & ENHANCEMENT
BW_sobel = edge(grayBird, 'sobel');
se = strel('disk', 1);
BW_sobel_dilated = imdilate(BW_sobel, se);

%% 5) K-MEANS CLUSTERING
numClusters = 3; 
[m, n, c] = size(birdImg);

% Reshape image into a list of pixels (convert to double for kmeans)
pixelData = double(reshape(birdImg, [], c));

% Perform k-means clustering
[idx, ~] = kmeans(pixelData, numClusters, ...
    'Distance','sqEuclidean','Replicates',3);

% Reshape cluster labels to image
pixelLabels = reshape(idx, [m, n]);

% Remove small noisy regions in each cluster
cleanedLabels = zeros(size(pixelLabels));
for i = 1:numClusters
    clusterMask = (pixelLabels == i);
    clusterMask = bwareaopen(clusterMask, 50);
    cleanedLabels(clusterMask) = i;
end
pixelLabels = cleanedLabels;

% Create a color-coded cluster image
clusteredImg = zeros(m, n, 3, 'uint8'); 
% We'll map each cluster to a color from the 'lines' colormap:
colors = uint8(255 * lines(numClusters)); % Nx3 array of RGB in [0,255]
for i = 1:numClusters
    mask = (pixelLabels == i);
    for ch = 1:3
        channelData = clusteredImg(:,:,ch);
        channelData(mask) = colors(i,ch);
        clusteredImg(:,:,ch) = channelData;
    end
end

%% 6) OBJECT DETECTION (Connected Components)
CC = bwconncomp(BW_rgb);
stats = regionprops(CC, 'Area', 'Centroid', 'BoundingBox');

% Create an overlay image showing bounding boxes & centroids
objectDetectionOverlay = birdImg;
for k = 1:length(stats)
    BB = round(stats(k).BoundingBox);
    x1 = BB(1); y1 = BB(2); w = BB(3); h = BB(4);
    x2 = x1 + w - 1; y2 = y1 + h - 1;

    % Clamp coordinates to image boundaries
    x2 = min(x2, size(objectDetectionOverlay,2));
    y2 = min(y2, size(objectDetectionOverlay,1));

    % Draw a green rectangle
    objectDetectionOverlay(y1:y2, [x1 x2], 1) = 0;
    objectDetectionOverlay(y1:y2, [x1 x2], 2) = 255;
    objectDetectionOverlay(y1:y2, [x1 x2], 3) = 0;
    objectDetectionOverlay([y1 y2], x1:x2, 1) = 0;
    objectDetectionOverlay([y1 y2], x1:x2, 2) = 255;
    objectDetectionOverlay([y1 y2], x1:x2, 3) = 0;

    % Draw a red cross at the centroid
    cx = round(stats(k).Centroid(1));
    cy = round(stats(k).Centroid(2));
    crossSize = 5;
    xs = max(1, cx-crossSize):min(size(objectDetectionOverlay,2), cx+crossSize);
    ys = max(1, cy-crossSize):min(size(objectDetectionOverlay,1), cy+crossSize);

    objectDetectionOverlay(cy, xs, 1) = 255;
    objectDetectionOverlay(cy, xs, 2) = 0;
    objectDetectionOverlay(cy, xs, 3) = 0;
    objectDetectionOverlay(ys, cx, 1) = 255;
    objectDetectionOverlay(ys, cx, 2) = 0;
    objectDetectionOverlay(ys, cx, 3) = 0;
end

%% 7) TURN THE BIRDS RED
% Create a copy of the original image, and wherever BW_rgb is true,
% set the pixel to bright red (R=255, G=0, B=0).
birdsRedOverlay = birdImg;
Rchan = birdsRedOverlay(:,:,1);
Gchan = birdsRedOverlay(:,:,2);
Bchan = birdsRedOverlay(:,:,3);

Rchan(BW_rgb) = 255;
Gchan(BW_rgb) = 0;
Bchan(BW_rgb) = 0;

birdsRedOverlay(:,:,1) = Rchan;
birdsRedOverlay(:,:,2) = Gchan;
birdsRedOverlay(:,:,3) = Bchan;

%% 8) SINGLE-FIGURE VISUALIZATION (10 SUBPLOTS)
figure('Name','Bird Image Analysis (Single Figure)',...
       'Position',[100 100 1400 900]);

% (1) Original Image
subplot(3,4,1);
imshow(birdImg);
title('Original Image');

% (2) HSV Image
subplot(3,4,2);
imshow(hsvBird);
title('HSV Image');

% (3) YCbCr Image
subplot(3,4,3);
imshow(ycbcrBird);
title('YCbCr Image');

% (4) Sobel Edges
subplot(3,4,4);
imshow(BW_sobel);
title('Sobel Edges');

% (5) Enhanced Edges (Dilated)
subplot(3,4,5);
imshow(BW_sobel_dilated);
title('Enhanced Edges');

% (6) Color Segmentation Mask
subplot(3,4,6);
imshow(BW_rgb);
title('Color Segmentation Mask');

% (7) Segmented Overlay
subplot(3,4,7);
imshow(colorSegmentedOverlay);
title('Segmented Overlay');

% (8) K-means Segmentation
subplot(3,4,8);
imshow(clusteredImg);
title('K-means Segmentation');

% (9) Object Detection Overlay
subplot(3,4,9);
imshow(objectDetectionOverlay);
title('Object Detection');

% (10) Birds Turned Red
subplot(3,4,10);
imshow(birdsRedOverlay);
title('Birds Turned Red');

