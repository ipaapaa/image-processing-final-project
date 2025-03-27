%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Image Processing Final Project
% Submitted by: Marc Olata and Job Isaac Ong (TN36)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% IMAGE 1

% 1) READ & RESIZE IMAGE
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



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Refined Image Segmentation and Object Detection in MATLAB
% Using a Single Figure for Visualization (Object of Interest: Dog)
%
% File: doggo.jpg
% GOAL:
%   Focus segmentation on the dog and avoid classifying the grassy background
%   as part of the dog. Then turn the dog's color to yellow instead of red.
%
% STEPS:
%   1) Preprocessing (resize to 512x512)
%   2) Color Space Conversions (RGB, HSV, YCbCr)
%   3) Refined Color Segmentation (Thresholding + Morphological Cleanup)
%      -> Keep only the largest connected component (assumes dog is largest)
%   4) Edge Detection (Sobel) + Enhancement (Dilation)
%   5) K-means Clustering (with small-object removal)
%   6) Object Detection (Connected Components)
%   7) Changing the Dog's Color to Yellow
%   8) Single-Figure Visualization (10 subplots)
%
% NOTE:
%   - Thresholds and morphological parameters here are illustrative.
%     You may need to tune them further for your specific image.
%   - Ensure "doggo.jpg" is in the current folder or MATLAB path.
%   - Everything is contained in one script (no separate functions).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

close all; clear; clc;

%% 1) READ & RESIZE IMAGE
originalImg = imread("dog.jpg");
dogImg = imresize(originalImg, [512 512]);
grayDog = rgb2gray(dogImg);  % for edge detection

%% 2) COLOR SPACE CONVERSIONS
hsvDog   = rgb2hsv(dogImg);
ycbcrDog = rgb2ycbcr(dogImg);

%% 3) REFINED COLOR SEGMENTATION
% We want to exclude green grass. A simple approach is to detect regions
% where the Green channel is significantly higher than Red & Blue, and remove them.

% Extract RGB channels
R = dogImg(:,:,1);
G = dogImg(:,:,2);
B = dogImg(:,:,3);

% Create a mask that *excludes* strong green pixels
% Example threshold: G > R + 30 and G > B + 30 => likely grass
grassMask = (G > R + 2) & (G > B + 2);

% Invert the mask so we keep regions that are NOT strong green
BW_rgb = ~grassMask;

% Morphological cleanup
BW_rgb = imclose(BW_rgb, strel('disk', 10));  % close small gaps
BW_rgb = imfill(BW_rgb, 'holes');            % fill holes
BW_rgb = bwareaopen(BW_rgb, 500);            % remove small objects

% Keep only the largest connected component (assumes the dog is largest)
CC_all = bwconncomp(BW_rgb);
if CC_all.NumObjects > 0
    statsAll = regionprops(CC_all, 'Area');
    [~, maxIdx] = max([statsAll.Area]);
    BW_rgb = (labelmatrix(CC_all) == maxIdx);
end

% Create an overlay image showing only the dog
colorSegmentedOverlay = dogImg;
colorSegmentedOverlay(repmat(~BW_rgb, [1 1 3])) = 0;

%% 4) EDGE DETECTION & ENHANCEMENT
BW_sobel = edge(grayDog, 'sobel');
se = strel('disk', 1);
BW_sobel_dilated = imdilate(BW_sobel, se);

%% 5) K-MEANS CLUSTERING
numClusters = 3; 
[m, n, c] = size(dogImg);

% Reshape image into a list of pixels (convert to double for kmeans)
pixelData = double(reshape(dogImg, [], c));

% Perform k-means clustering
[idx, ~] = kmeans(pixelData, numClusters, ...
    'Distance','sqEuclidean','Replicates',3);

% Reshape cluster labels to the image
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
colors = uint8(255 * lines(numClusters)); % Nx3 array of RGB in [0,255]
for i = 1:numClusters
    mask = (pixelLabels == i);
    for ch = 1:3
        channelData = clusteredImg(:,:,ch);
        channelData(mask) = colors(i,ch);
        clusteredImg(:,:,ch) = channelData;
    end
end

%% 6) OBJECT DETECTION (Connected Components on the refined dog mask)
CC = bwconncomp(BW_rgb);
stats = regionprops(CC, 'Area', 'Centroid', 'BoundingBox');

% Create an overlay image showing bounding boxes & centroids
objectDetectionOverlay = dogImg;
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

%% 7) CHANGE DOG'S COLOR TO YELLOW (WITH REDUCED OPACITY)
% Instead of replacing the dog region entirely with yellow, we blend it
% with the original image using alpha blending.
alpha = 0.7;  % 0 = no yellow overlay, 1 = full yellow (adjust as needed)
% Convert the image to double precision for blending
dogYellowOverlay = im2double(dogImg);

% Create a yellow overlay image (yellow = [1 1 0] in normalized RGB)
yellowOverlay = ones(size(dogImg));
yellowOverlay(:,:,1) = 1;
yellowOverlay(:,:,2) = 1;
yellowOverlay(:,:,3) = 0;

% Create a mask for the dog region (replicate mask to three channels)
mask = repmat(BW_rgb, [1 1 3]);

% Blend the yellow overlay with the original image on the dog's region
dogYellowOverlay(mask) = alpha * yellowOverlay(mask) + (1 - alpha) * dogYellowOverlay(mask);
dogYellowOverlay = im2uint8(dogYellowOverlay);

%% 8) SINGLE-FIGURE VISUALIZATION (10 SUBPLOTS)
figure('Name','Refined Doggo Image Analysis (Single Figure)',...
       'Position',[100 100 1400 900]);

% (1) Original Image
subplot(3,4,1);
imshow(dogImg);
title('Original Image');

% (2) HSV Image
subplot(3,4,2);
imshow(hsvDog);
title('HSV Image');

% (3) YCbCr Image
subplot(3,4,3);
imshow(ycbcrDog);
title('YCbCr Image');

% (4) Sobel Edges
subplot(3,4,4);
imshow(BW_sobel);
title('Sobel Edges');

% (5) Enhanced Edges (Dilated)
subplot(3,4,5);
imshow(BW_sobel_dilated);
title('Enhanced Edges');

% (6) Refined Color Segmentation Mask
subplot(3,4,6);
imshow(BW_rgb);
title('Refined Color Mask');

% (7) Segmented Overlay (Dog Only)
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

% (10) Dog Turned Yellow
subplot(3,4,10);
imshow(dogYellowOverlay);
title('Dog Turned Yellow');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% END OF SCRIPT
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
