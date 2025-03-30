%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Image Processing Final Project
% Submitted by: Marc Olata and Job Isaac Ong (TN36)
%
% This script processes three images:
%   1) Bird Image ("birdies.jpg")
%       - Outputs: Original, HSV, YCbCr, Sobel/Prewitt edges, enhanced edges,
%         color segmentation mask, segmented overlay, K-means segmentation,
%         object detection overlay (thicker boxes), birds turned red, and background blur.
%
%   2) Dog Image ("doggo.jpg")
%       - Similar outputs as for birds, with the dog turned yellow (with partial opacity).
%
%   3) Landscape Image ("landscape.jpg")
%       - In addition to the common outputs (original, HSV, YCbCr, edge maps, k-means segmentation, background blur),
%         this section includes a more defined (manual) color segmentation (e.g., sky, vegetation, other)
%         and overlays for both the defined segmentation and k-means segmentation.
%
% Each image is processed separately and its results are displayed in a separate figure.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% -------------------- BIRD IMAGE ANALYSIS -----------s---------
close all; clear; clc;
set(0, 'DefaultFigureWindowStyle', 'normal');
set(0, 'DefaultFigurePosition', [100, 50, 1000, 700]);
%--- BIRD IMAGE ---
originalImg = imread('birdies.jpg');
birdImg = imresize(originalImg, [512 512]);
grayBird = rgb2gray(birdImg);

% COLOR SPACE CONVERSIONS
hsvBird   = rgb2hsv(birdImg);
ycbcrBird = rgb2ycbcr(birdImg);

% COLOR SEGMENTATION (RGB-based): Threshold dark pixels (silhouettes)
R = birdImg(:,:,1);
G = birdImg(:,:,2);
B = birdImg(:,:,3);
darkThreshold = 70;
BW_bird = (R < darkThreshold) & (G < darkThreshold) & (B < darkThreshold);
BW_bird = bwareaopen(BW_bird, 50);
colorSegmentedOverlayBird = birdImg;
colorSegmentedOverlayBird(repmat(~BW_bird, [1 1 3])) = 0;

% EDGE DETECTION
BW_sobel_bird   = edge(grayBird, 'sobel');
BW_prewitt_bird = edge(grayBird, 'prewitt');
se = strel('disk', 1);
BW_sobel_dilated_bird = imdilate(BW_sobel_bird, se);

% K-MEANS CLUSTERING
numClusters = 3;
[m, n, c] = size(birdImg);
pixelData = double(reshape(birdImg, [], c));
[idx, ~] = kmeans(pixelData, numClusters, 'Distance','sqEuclidean','Replicates',3);
pixelLabels = reshape(idx, [m, n]);
cleanedLabels = zeros(size(pixelLabels));
for i = 1:numClusters
    clusterMask = (pixelLabels == i);
    clusterMask = bwareaopen(clusterMask, 50);
    cleanedLabels(clusterMask) = i;
end
pixelLabels = cleanedLabels;
clusteredImgBird = zeros(m, n, 3, 'uint8');
colors = uint8(255 * lines(numClusters));
for i = 1:numClusters
    mask = (pixelLabels == i);
    for ch = 1:3
        temp = clusteredImgBird(:,:,ch);
        temp(mask) = colors(i,ch);
        clusteredImgBird(:,:,ch) = temp;
    end
end

% OBJECT DETECTION (Thicker bounding boxes via insertShape)
CC = bwconncomp(BW_bird);
stats = regionprops(CC, 'BoundingBox', 'Centroid');
rectangles = [];
for k = 1:length(stats)
    rectangles = [rectangles; stats(k).BoundingBox];
end
objectDetectionOverlayBird = birdImg;
if ~isempty(rectangles)
    objectDetectionOverlayBird = insertShape(birdImg, 'Rectangle', rectangles, 'Color', 'green', 'LineWidth', 3);
end
centroids = [];
for k = 1:length(stats)
    centroids = [centroids; stats(k).Centroid];
end
if ~isempty(centroids)
    objectDetectionOverlayBird = insertMarker(objectDetectionOverlayBird, centroids, 'x', 'Color', 'red', 'Size', 10);
end

% TURN THE BIRDS RED (Full replacement on mask)
birdsRedOverlay = birdImg;
Rchan = birdsRedOverlay(:,:,1);
Gchan = birdsRedOverlay(:,:,2);
Bchan = birdsRedOverlay(:,:,3);
Rchan(BW_bird) = 255;
Gchan(BW_bird) = 0;
Bchan(BW_bird) = 0;
birdsRedOverlay(:,:,1) = Rchan;
birdsRedOverlay(:,:,2) = Gchan;
birdsRedOverlay(:,:,3) = Bchan;

% BACKGROUND BLURRING
blurredBird = imgaussfilt(birdImg, 10);
birdBackgroundBlurred = birdImg;
birdBackgroundBlurred(~repmat(BW_bird, [1 1 3])) = blurredBird(~repmat(BW_bird, [1 1 3]));

% DISPLAY: 12 subplots for Bird Analysis
figure('Name','Bird Image Analysis','Position',[50 50 1000 700]);
subplot(3,4,1), imshow(birdImg), title('Original Image');
subplot(3,4,2), imshow(hsvBird), title('HSV Image');
subplot(3,4,3), imshow(ycbcrBird), title('YCbCr Image');
subplot(3,4,4), imshow(BW_sobel_bird), title('Sobel Edges');
subplot(3,4,5), imshow(BW_prewitt_bird), title('Prewitt Edges');
subplot(3,4,6), imshow(BW_sobel_dilated_bird), title('Enhanced Edges');
subplot(3,4,7), imshow(BW_bird), title('Color Segmentation Mask');
subplot(3,4,8), imshow(colorSegmentedOverlayBird), title('Segmented Overlay');
subplot(3,4,9), imshow(clusteredImgBird), title('K-means Segmentation');
subplot(3,4,10), imshow(objectDetectionOverlayBird), title('Object Detection');
subplot(3,4,11), imshow(birdsRedOverlay), title('Birds Turned Red');
subplot(3,4,12), imshow(birdBackgroundBlurred), title('Background Blurred');

%% -------------------- DOG IMAGE ANALYSIS --------------------
%--- DOG IMAGE ---
originalDog = imread('dog.jpg');  % or "dog.jpg" as appropriate
dogImg = imresize(originalDog, [512 512]);
grayDog = rgb2gray(dogImg);

% COLOR SPACE CONVERSIONS
hsvDog   = rgb2hsv(dogImg);
ycbcrDog = rgb2ycbcr(dogImg);

% REFINED COLOR SEGMENTATION: Exclude strongly green (grass)
R = dogImg(:,:,1);
G = dogImg(:,:,2);
B = dogImg(:,:,3);
grassMask = (G > R + 2) & (G > B + 2);
BW_dog = ~grassMask;
BW_dog = imclose(BW_dog, strel('disk',10));
BW_dog = imfill(BW_dog, 'holes');
BW_dog = bwareaopen(BW_dog, 500);
CC_all = bwconncomp(BW_dog);
if CC_all.NumObjects > 0
    statsAll = regionprops(CC_all, 'Area');
    [~, maxIdx] = max([statsAll.Area]);
    BW_dog = (labelmatrix(CC_all)==maxIdx);
end
colorSegmentedOverlayDog = dogImg;
colorSegmentedOverlayDog(repmat(~BW_dog, [1 1 3])) = 0;

% EDGE DETECTION
BW_sobel_dog   = edge(grayDog, 'sobel');
BW_prewitt_dog = edge(grayDog, 'prewitt');
se = strel('disk',1);
BW_sobel_dog_dilated = imdilate(BW_sobel_dog, se);

% K-MEANS CLUSTERING
numClusters = 3;
[m, n, c] = size(dogImg);
pixelData = double(reshape(dogImg, [], c));
[idx, ~] = kmeans(pixelData, numClusters, 'Distance','sqEuclidean','Replicates',3);
pixelLabels = reshape(idx, [m, n]);
cleanedLabels = zeros(size(pixelLabels));
for i = 1:numClusters
    clusterMask = (pixelLabels == i);
    clusterMask = bwareaopen(clusterMask, 50);
    cleanedLabels(clusterMask) = i;
end
pixelLabels = cleanedLabels;
clusteredImgDog = zeros(m, n, 3, 'uint8');
colors = uint8(255 * lines(numClusters));
for i = 1:numClusters
    mask = (pixelLabels == i);
    for ch = 1:3
        temp = clusteredImgDog(:,:,ch);
        temp(mask) = colors(i,ch);
        clusteredImgDog(:,:,ch) = temp;
    end
end

% OBJECT DETECTION (Thicker bounding boxes)
CC = bwconncomp(BW_dog);
stats = regionprops(CC, 'BoundingBox', 'Centroid');
rectangles = [];
for k = 1:length(stats)
    rectangles = [rectangles; stats(k).BoundingBox];
end
objectDetectionOverlayDog = dogImg;
if ~isempty(rectangles)
    objectDetectionOverlayDog = insertShape(dogImg, 'Rectangle', rectangles, 'Color', 'green', 'LineWidth', 3);
end
centroids = [];
for k = 1:length(stats)
    centroids = [centroids; stats(k).Centroid];
end
if ~isempty(centroids)
    objectDetectionOverlayDog = insertMarker(objectDetectionOverlayDog, centroids, 'x', 'Color', 'red', 'Size', 10);
end

% TURN THE DOG YELLOW (with partial opacity)
alpha = 0.7;
dogYellowOverlay = im2double(dogImg);
yellowOverlay = ones(size(dogImg));
yellowOverlay(:,:,1) = 1; yellowOverlay(:,:,2) = 1; yellowOverlay(:,:,3) = 0;
mask = repmat(BW_dog, [1 1 3]);
dogYellowOverlay(mask) = alpha * yellowOverlay(mask) + (1 - alpha) * dogYellowOverlay(mask);
dogYellowOverlay = im2uint8(dogYellowOverlay);

% BACKGROUND BLURRING
blurredDog = imgaussfilt(dogImg, 10);
dogBackgroundBlurred = dogImg;
dogBackgroundBlurred(~repmat(BW_dog, [1 1 3])) = blurredDog(~repmat(BW_dog, [1 1 3]));

% DISPLAY: 12 subplots for Dog Analysis
figure('Name','Dog Image Analysis','Position',[55 55 1000 700]);
subplot(3,4,1), imshow(dogImg), title('Original Image');
subplot(3,4,2), imshow(hsvDog), title('HSV Image');
subplot(3,4,3), imshow(ycbcrDog), title('YCbCr Image');
subplot(3,4,4), imshow(BW_sobel_dog), title('Sobel Edges');
subplot(3,4,5), imshow(BW_prewitt_dog), title('Prewitt Edges');
subplot(3,4,6), imshow(BW_sobel_dog_dilated), title('Enhanced Edges');
subplot(3,4,7), imshow(BW_dog), title('Refined Color Mask');
subplot(3,4,8), imshow(colorSegmentedOverlayDog), title('Segmented Overlay');
subplot(3,4,9), imshow(clusteredImgDog), title('K-means Segmentation');
subplot(3,4,10), imshow(objectDetectionOverlayDog), title('Object Detection');
subplot(3,4,11), imshow(dogYellowOverlay), title('Dog Turned Yellow');
subplot(3,4,12), imshow(dogBackgroundBlurred), title('Background Blurred');

%% -------------------- LANDSCAPE IMAGE ANALYSIS --------------------
%--- LANDSCAPE IMAGE ---
landscape = imread('landscape.jpg');
% Optionally resize if desired:
landscape = imresize(landscape, [512 512]);

% COLOR SPACE CONVERSIONS
hsvLand = rgb2hsv(landscape);
ycbcrLand = rgb2ycbcr(landscape);

% EDGE DETECTION
grayLand = rgb2gray(landscape);
BW_sobel_land = edge(grayLand, 'sobel');
BW_prewitt_land = edge(grayLand, 'prewitt');
se = strel('disk', 1);
BW_sobel_dilated_land = imdilate(BW_sobel_land, se);

% (A) Defined Color Segmentation in HSV
% Example thresholds for a typical landscape:
%   Let's assume: sky is blue (H between 0.5 and 0.7, low S), vegetation is green (H between 0.20 and 0.40)
skyMask = (hsvLand(:,:,1) >= 0.5 & hsvLand(:,:,1) <= 0.7) & (hsvLand(:,:,2) <= 0.3) & (hsvLand(:,:,3) >= 0.5);
vegMask = (hsvLand(:,:,1) >= 0.20 & hsvLand(:,:,1) <= 0.40) & (hsvLand(:,:,2) >= 0.2) & (hsvLand(:,:,3) >= 0.2);
labelsDefined = zeros(size(skyMask)); 
labelsDefined(skyMask) = 1;   % label 1 = sky
labelsDefined(vegMask) = 2;   % label 2 = vegetation
% The rest remain label 0 (other)

% Build a defined color segmentation overlay:
segDefined = zeros(size(landscape), 'uint8');
% Map: 0-> [255,0,0] (red for other), 1-> [0,0,255] (blue for sky), 2-> [0,255,0] (green for vegetation)
colorMap = [255 0 0; 0 0 255; 0 255 0];
for lbl = 0:2
    mask = (labelsDefined == lbl);
    for ch = 1:3
        segDefined(:,:,ch) = segDefined(:,:,ch) + uint8(mask)*colorMap(lbl+1,ch);
    end
end

% (B) K-means Clustering on Landscape
[mL, nL, cL] = size(landscape);
imgDouble = im2double(landscape);
pixelDataL = reshape(imgDouble, [mL*nL, cL]);
numClustersLand = 4;
[idxL, ~] = kmeans(pixelDataL, numClustersLand, 'Distance','sqEuclidean','Replicates',3);
pixelLabelsL = reshape(idxL, [mL, nL]);
cleanedLabelsL = zeros(size(pixelLabelsL));
for i = 1:numClustersLand
    clusterMask = (pixelLabelsL == i);
    clusterMask = bwareaopen(clusterMask, 50);
    cleanedLabelsL(clusterMask) = i;
end
pixelLabelsL = cleanedLabelsL;
clusteredImgLand = zeros(mL, nL, 3, 'uint8');
clusterColors = uint8(255 * lines(numClustersLand));
for i = 1:numClustersLand
    mask = (pixelLabelsL == i);
    for ch = 1:3
        temp = clusteredImgLand(:,:,ch);
        temp(mask) = clusterColors(i,ch);
        clusteredImgLand(:,:,ch) = temp;
    end
end

% (C) Overlays: Blend defined segmentation and K-means with original
alpha = 0.5;
overlayDefined = im2double(landscape);
for ch = 1:3
    overlayDefined(:,:,ch) = (1 - alpha)*overlayDefined(:,:,ch) + alpha*(im2double(segDefined(:,:,ch)));
end
overlayDefined = im2uint8(overlayDefined);

overlayKmeans = im2double(landscape);
for ch = 1:3
    overlayKmeans(:,:,ch) = (1 - alpha)*overlayKmeans(:,:,ch) + alpha*(im2double(clusteredImgLand(:,:,ch)));
end
overlayKmeans = im2uint8(overlayKmeans);

% BACKGROUND BLURRING for Landscape
blurredLand = imgaussfilt(landscape, 10);
% Blend the original and blurred image with an alpha factor of 0.5:
landBackgroundBlurred = im2uint8((im2double(landscape) + im2double(blurredLand)) / 2);

% If imblend is not available, we can do:
landBackgroundBlurred = im2uint8((im2double(landscape) + im2double(blurredLand))/2);

% DISPLAY: 12 subplots for Landscape Analysis
figure('Name','Landscape Image Analysis','Position',[60 60 1000 700]);
subplot(3,4,1), imshow(landscape), title('Original Image');
subplot(3,4,2), imshow(hsvLand), title('HSV Image');
subplot(3,4,3), imshow(ycbcrLand), title('YCbCr Image');
subplot(3,4,4), imshow(BW_sobel_land), title('Sobel Edges');
subplot(3,4,5), imshow(BW_prewitt_land), title('Prewitt Edges');
subplot(3,4,6), imshow(BW_sobel_dilated_land), title('Enhanced Edges');
subplot(3,4,7), imshow(segDefined), title('Defined Color Segmentation');
subplot(3,4,8), imshow(overlayDefined), title('Overlay: Original + Defined');
subplot(3,4,9), imshow(clusteredImgLand), title('K-means Segmentation');
subplot(3,4,10), imshow(overlayKmeans), title('Overlay: Original + K-means');
subplot(3,4,11), imshow(landBackgroundBlurred), title('Background Blurred');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% END OF SCRIPT
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Car Image Analysis (Top-Hat + Shape Filtering)
% 
% 1) Read & Resize
% 2) Top-Hat Filtering
% 3) Threshold + Morphological Cleanup
% 4) Shape-Based Filtering (Remove elongated, thin objects)
% 5) Edge Detection (Sobel, Prewitt) + Dilation
% 6) K-means Clustering
% 7) Object Detection + Thicker Boxes
% 8) Cars Turned Red
% 9) Background Blurring
% 10) Single-Figure Visualization
%
% NOTE:
%   - Adjust 'seSize', 'thVal', morphological parameters, and 
%     shape-based filtering thresholds for best results.
%   - Ensure "car.jpg" is in your working directory or path.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

close all; clear; clc;

%% 1) READ & RESIZE
originalCar = imread('car.jpg');
carImg = imresize(originalCar, [512 512]);

grayCar = rgb2gray(carImg);

%% 2) TOP-HAT FILTERING
% The top-hat operation highlights objects brighter than the local background.
% Increase 'seSize' if cars are large or the road is more uneven.
seSize = 20;  
se = strel('disk', seSize);

topHatImg = imtophat(grayCar, se);
topHatImg = imadjust(topHatImg);  % enhance contrast

%% 3) THRESHOLD & MORPHOLOGICAL CLEANUP
% Adjust 'thVal' in [0,1] for your image
thVal = 0.15;  
BW_car = imbinarize(im2double(topHatImg), thVal);

% Merge and clean up car regions
BW_car = imclose(BW_car, strel('disk', 5));
BW_car = imfill(BW_car, 'holes');
BW_car = bwareaopen(BW_car, 200);

%% 4) SHAPE-BASED FILTERING
% Remove elongated, thin objects (e.g., lane markings) by checking aspect ratio
% For each region, compute MajorAxisLength, MinorAxisLength. 
% If aspect ratio is too large (e.g., >5), discard it.

CC = bwconncomp(BW_car);
statsAll = regionprops(CC, 'Area', 'MajorAxisLength', 'MinorAxisLength', 'PixelIdxList');

BW_clean = false(size(BW_car));
for k = 1:length(statsAll)
    maj = statsAll(k).MajorAxisLength;
    minr = statsAll(k).MinorAxisLength;
    aspectRatio = maj / max(minr,1);  % avoid dividing by zero

    % Example thresholds:
    %   - Keep area > 200
    %   - Keep aspect ratio < 5 (tweak as needed)
    if statsAll(k).Area > 200 && aspectRatio < 5
        BW_clean(statsAll(k).PixelIdxList) = true;
    end
end
BW_car = BW_clean;

% Create overlay image showing only car regions
colorSegmentedOverlayCar = carImg;
colorSegmentedOverlayCar(repmat(~BW_car, [1 1 3])) = 0;

%% 5) EDGE DETECTION & ENHANCEMENT
BW_sobel_car   = edge(grayCar, 'sobel');
BW_prewitt_car = edge(grayCar, 'prewitt');

BW_sobel_car_dilated = imdilate(BW_sobel_car, strel('disk', 1));

%% 6) K-MEANS CLUSTERING
numClusters = 3;
[m, n, c] = size(carImg);
pixelData = double(reshape(carImg, [], c));

[idx, ~] = kmeans(pixelData, numClusters, 'Distance','sqEuclidean','Replicates',3);
pixelLabels = reshape(idx, [m, n]);

% Remove small noisy regions
cleanedLabels = zeros(size(pixelLabels));
for i = 1:numClusters
    clusterMask = (pixelLabels == i);
    clusterMask = bwareaopen(clusterMask, 50);
    cleanedLabels(clusterMask) = i;
end
pixelLabels = cleanedLabels;

% Build color-coded cluster image
clusteredImgCar = zeros(m, n, 3, 'uint8');
colors = uint8(255 * lines(numClusters));
for i = 1:numClusters
    mask = (pixelLabels == i);
    for ch = 1:3
        tmp = clusteredImgCar(:,:,ch);
        tmp(mask) = colors(i,ch);
        clusteredImgCar(:,:,ch) = tmp;
    end
end

%% 7) OBJECT DETECTION (Connected Components)
CC = bwconncomp(BW_car);
stats = regionprops(CC, 'BoundingBox', 'Centroid');

rectangles = [];
for k = 1:length(stats)
    rectangles = [rectangles; stats(k).BoundingBox];
end

objectDetectionOverlayCar = carImg;
if ~isempty(rectangles)
    objectDetectionOverlayCar = insertShape(carImg, 'Rectangle', rectangles, ...
        'Color', 'green', 'LineWidth', 3);
end

% Mark centroids with red crosses
centroids = [];
for k = 1:length(stats)
    centroids = [centroids; stats(k).Centroid];
end
if ~isempty(centroids)
    objectDetectionOverlayCar = insertMarker(objectDetectionOverlayCar, centroids, ...
        'x', 'Color', 'red', 'Size', 10);
end

%% 8) CARS TURNED RED
carsRedOverlay = carImg;
Rchan = carsRedOverlay(:,:,1);
Gchan = carsRedOverlay(:,:,2);
Bchan = carsRedOverlay(:,:,3);

Rchan(BW_car) = 255;
Gchan(BW_car) = 0;
Bchan(BW_car) = 0;

carsRedOverlay(:,:,1) = Rchan;
carsRedOverlay(:,:,2) = Gchan;
carsRedOverlay(:,:,3) = Bchan;

%% 9) BACKGROUND BLURRING
blurredCar = imgaussfilt(carImg, 10);
carBackgroundBlurred = carImg;
carBackgroundBlurred(~repmat(BW_car, [1 1 3])) = blurredCar(~repmat(BW_car, [1 1 3]));

%% 10) SINGLE-FIGURE VISUALIZATION (12 Subplots)
figure('Name','Car Image Analysis','Position',[65 65 1000 700]);

% (1) Original
subplot(3,4,1);
imshow(carImg);
title('Original Image');

% (2) Top-Hat Result
subplot(3,4,2);
imshow(topHatImg, []);
title('Top-Hat Result');

% (3) Mask (after shape filtering)
subplot(3,4,3);
imshow(BW_car);
title('Car Mask (Shape Filtered)');

% (4) Segmented Overlay
subplot(3,4,4);
imshow(colorSegmentedOverlayCar);
title('Segmented Overlay');

% (5) Sobel Edges
subplot(3,4,5);
imshow(BW_sobel_car);
title('Sobel Edges');

% (6) Prewitt Edges
subplot(3,4,6);
imshow(BW_prewitt_car);
title('Prewitt Edges');

% (7) Enhanced Edges (Dilated)
subplot(3,4,7);
imshow(BW_sobel_car_dilated);
title('Enhanced Edges');

% (8) K-means Segmentation
subplot(3,4,8);
imshow(clusteredImgCar);
title('K-means Segmentation');

% (9) Object Detection
subplot(3,4,9);
imshow(objectDetectionOverlayCar);
title('Object Detection');

% (10) Cars Turned Red
subplot(3,4,10);
imshow(carsRedOverlay);
title('Cars Turned Red');

% (11) Background Blurred
subplot(3,4,11);
imshow(carBackgroundBlurred);
title('Background Blurred');

% (12) Original or any extra result
subplot(3,4,12);
imshow(carImg);
title('End of Demo');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% END OF SCRIPT
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Fruit Image Analysis (Refined for Pear Only)
% Object of interest: The yellow pear
%
% Processes:
%   1) Preprocessing (read, resize)
%   2) Color Space Conversions (RGB->HSV, YCbCr)
%   3) Refined Color Segmentation (Narrower HSV range + keep only largest region)
%   4) Edge Detection (Sobel, Prewitt) + Enhancement (dilation)
%   5) K-means Clustering
%   6) Object Detection (Connected Components) + Thicker bounding boxes
%   7) Turning the pear red
%   8) Background Blurring
%   9) Single-Figure Visualization (12 subplots)
%
% NOTE:
%   - Adjust the HSV thresholds for your pear's color.
%   - If needed, tweak morphological sizes and the "largest region" logic.
%   - Ensure "fruit.jpg" (or your actual filename) is in your path.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

close all; clear; clc;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Fruit Image Analysis (Further Refined for the Yellow Pear)
% Object of interest: The yellow pear
%
% 1) Preprocessing (read, resize)
% 2) Color Space Conversions (RGB->HSV, YCbCr)
% 3) Narrow HSV Threshold + Largest Connected Component
% 4) Edge Detection (Sobel, Prewitt) + Dilation
% 5) K-means Clustering
% 6) Object Detection (Connected Components) + Thicker boxes
% 7) Turn the Pear Red
% 8) Background Blurring
% 9) Single-Figure Visualization (12 subplots)
%
% NOTE:
%   - Adjust the hue, saturation, value ranges, morphological sizes, and
%     area thresholds to precisely isolate the pear in your image.
%   - Ensure "fruit.jpg" (or your actual filename) is in your path.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

close all; clear; clc;

%% 1) READ & RESIZE IMAGE
originalImg = imread('fruits.jpg');  % Replace with your actual filename
fruitImg = imresize(originalImg, [512 512]);
grayFruit = rgb2gray(fruitImg);

%% 2) COLOR SPACE CONVERSIONS
hsvFruit   = rgb2hsv(fruitImg);
ycbcrFruit = rgb2ycbcr(fruitImg);

%% 3) REFINED COLOR SEGMENTATION FOR THE PEAR
% Try slightly tighter hue bounds. If the pear is more/less orange or greenish,
% shift these ranges accordingly.
H = hsvFruit(:,:,1);  % Hue   in [0,1]
S = hsvFruit(:,:,2);  % Sat   in [0,1]
V = hsvFruit(:,:,3);  % Value in [0,1]

% Updated narrower thresholds for a bright yellowish pear:
%   Hue ~ [0.12, 0.16]
%   Saturation >= 0.45
%   Value >= 0.5
BW_pear = (H >= 0.12 & H <= 0.16) & ...
          (S >= 0.45) & ...
          (V >= 0.5);

% Morphological cleanup
BW_pear = imclose(BW_pear, strel('disk', 5));  % unify pear region
BW_pear = imfill(BW_pear, 'holes');            % fill holes
BW_pear = bwareaopen(BW_pear, 50);             % remove small specks

% Keep only the largest connected component
CC_all = bwconncomp(BW_pear);
if CC_all.NumObjects > 0
    statsAll = regionprops(CC_all, 'Area', 'PixelIdxList');
    [~, maxIdx] = max([statsAll.Area]);
    BW_largest = false(size(BW_pear));
    BW_largest(statsAll(maxIdx).PixelIdxList) = true;
    BW_pear = BW_largest;
end

% Create an overlay image showing only the pear region
colorSegmentedOverlayPear = fruitImg;
colorSegmentedOverlayPear(repmat(~BW_pear, [1 1 3])) = 0;

%% 4) EDGE DETECTION & ENHANCEMENT
BW_sobel_fruit   = edge(grayFruit, 'sobel');
BW_prewitt_fruit = edge(grayFruit, 'prewitt');
BW_sobel_fruit_dilated = imdilate(BW_sobel_fruit, strel('disk', 1));

%% 5) K-MEANS CLUSTERING
numClusters = 3;
[m, n, c] = size(fruitImg);
pixelData = double(reshape(fruitImg, [], c));

[idx, ~] = kmeans(pixelData, numClusters, 'Distance','sqEuclidean','Replicates',3);
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
clusteredImgFruit = zeros(m, n, 3, 'uint8');
colors = uint8(255 * lines(numClusters));
for i = 1:numClusters
    mask = (pixelLabels == i);
    for ch = 1:3
        temp = clusteredImgFruit(:,:,ch);
        temp(mask) = colors(i,ch);
        clusteredImgFruit(:,:,ch) = temp;
    end
end

%% 6) OBJECT DETECTION (Connected Components)
CC = bwconncomp(BW_pear);
stats = regionprops(CC, 'BoundingBox', 'Centroid');

% Draw thicker bounding boxes
rectangles = [];
for k = 1:length(stats)
    rectangles = [rectangles; stats(k).BoundingBox];
end

objectDetectionOverlayPear = fruitImg;
if ~isempty(rectangles)
    objectDetectionOverlayPear = insertShape(fruitImg, 'Rectangle', rectangles, ...
        'Color', 'green', 'LineWidth', 3);
end

% Mark centroids with red crosses
centroids = [];
for k = 1:length(stats)
    centroids = [centroids; stats(k).Centroid];
end
if ~isempty(centroids)
    objectDetectionOverlayPear = insertMarker(objectDetectionOverlayPear, centroids, ...
        'x', 'Color', 'red', 'Size', 10);
end

%% 7) TURN THE PEAR RED
pearRedOverlay = fruitImg;
Rchan = pearRedOverlay(:,:,1);
Gchan = pearRedOverlay(:,:,2);
Bchan = pearRedOverlay(:,:,3);

Rchan(BW_pear) = 255;
Gchan(BW_pear) = 0;
Bchan(BW_pear) = 0;

pearRedOverlay(:,:,1) = Rchan;
pearRedOverlay(:,:,2) = Gchan;
pearRedOverlay(:,:,3) = Bchan;

%% 8) BACKGROUND BLURRING
blurredFruit = imgaussfilt(fruitImg, 10);
fruitBackgroundBlurred = fruitImg;
fruitBackgroundBlurred(~repmat(BW_pear, [1 1 3])) = ...
    blurredFruit(~repmat(BW_pear, [1 1 3]));

%% 9) SINGLE-FIGURE VISUALIZATION (12 Subplots)
figure('Name','Refined Fruit Image Analysis','Position',[65 65 1000 700]);

% (1) Original Image
subplot(3,4,1);
imshow(fruitImg);
title('Original Image');

% (2) HSV Image
subplot(3,4,2);
imshow(hsvFruit);
title('HSV Image');

% (3) YCbCr Image
subplot(3,4,3);
imshow(ycbcrFruit);
title('YCbCr Image');

% (4) Sobel Edges
subplot(3,4,4);
imshow(BW_sobel_fruit);
title('Sobel Edges');

% (5) Prewitt Edges
subplot(3,4,5);
imshow(BW_prewitt_fruit);
title('Prewitt Edges');

% (6) Enhanced Edges (Dilated)
subplot(3,4,6);
imshow(BW_sobel_fruit_dilated);
title('Enhanced Edges');

% (7) Refined Pear Mask
subplot(3,4,7);
imshow(BW_pear);
title('Refined Pear Mask');

% (8) Segmented Overlay (Pear Only)
subplot(3,4,8);
imshow(colorSegmentedOverlayPear);
title('Segmented Overlay');

% (9) K-means Segmentation
subplot(3,4,9);
imshow(clusteredImgFruit);
title('K-means Segmentation');

% (10) Object Detection
subplot(3,4,10);
imshow(objectDetectionOverlayPear);
title('Object Detection');

% (11) Pear Turned Red
subplot(3,4,11);
imshow(pearRedOverlay);
title('Pear Turned Red');

% (12) Background Blurred
subplot(3,4,12);
imshow(fruitBackgroundBlurred);
title('Background Blurred');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% END OF SCRIPT
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



