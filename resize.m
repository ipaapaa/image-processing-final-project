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
% Car Image Analysis (Top-Hat Approach for Difficult Road/Car Segmentation)
%
% 1) Preprocessing (read, resize)
% 2) Top-Hat Filtering in Grayscale
% 3) Threshold + Morphological Cleanup => Car Mask
% 4) Edge Detection (Sobel, Prewitt) + Dilation
% 5) K-means Clustering
% 6) Object Detection (Connected Components) + Thicker Boxes
% 7) Cars Turned Red (Overlay)
% 8) Background Blurring
% 9) Single-Figure Visualization
%
% NOTE:
%   - The top-hat filter size, threshold, and morphological steps
%     should be tuned for your specific road/car image.
%   - If cars are darker than the road, consider a bottom-hat approach
%     or invert the image.
%   - Ensure "car.jpg" is in your working directory or path.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

close all; clear; clc;

%% 1) READ & RESIZE
originalCar = imread('car.jpg');
carImg = imresize(originalCar, [512 512]);

% Convert to grayscale for top-hat
grayCar = rgb2gray(carImg);

%% 2) TOP-HAT FILTERING
% The top-hat operation extracts bright objects from an uneven background.
% Choose a structuring element size large enough to approximate the road's background.
seSize = 20;  % try adjusting between 15–30, or more, depending on car size
se = strel('disk', seSize);

% Perform top-hat: highlights regions brighter than the surrounding background
topHatImg = imtophat(grayCar, se);

% (Optional) Enhance contrast of topHatImg to make thresholding easier
topHatImg = imadjust(topHatImg);

%% 3) THRESHOLD & MORPHOLOGY
% Choose a threshold that captures the bright or midtone cars
% The exact value is image-dependent. Adjust as needed.
thVal = 0.15;  % in [0,1] after imadjust
BW_car = imbinarize(im2double(topHatImg), thVal);

% Morphological cleanup
BW_car = imclose(BW_car, strel('disk', 5));  % unify car regions
BW_car = imfill(BW_car, 'holes');           % fill holes inside cars
BW_car = bwareaopen(BW_car, 200);           % remove small objects

% Create a color-segmentation overlay (only show car regions)
colorSegmentedOverlayCar = carImg;
colorSegmentedOverlayCar(repmat(~BW_car, [1 1 3])) = 0;

%% 4) EDGE DETECTION & ENHANCEMENT
BW_sobel_car   = edge(grayCar, 'sobel');
BW_prewitt_car = edge(grayCar, 'prewitt');

% Dilation
seDilate = strel('disk', 1);
BW_sobel_car_dilated = imdilate(BW_sobel_car, seDilate);

%% 5) K-MEANS CLUSTERING
numClusters = 3;
[m, n, c] = size(carImg);
pixelData = double(reshape(carImg, [], c));

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
clusteredImgCar = zeros(m, n, 3, 'uint8');
colors = uint8(255 * lines(numClusters));
for i = 1:numClusters
    mask = (pixelLabels == i);
    for ch = 1:3
        temp = clusteredImgCar(:,:,ch);
        temp(mask) = colors(i,ch);
        clusteredImgCar(:,:,ch) = temp;
    end
end

%% 6) OBJECT DETECTION (Connected Components)
CC = bwconncomp(BW_car);
stats = regionprops(CC, 'BoundingBox', 'Centroid');

% Draw thicker bounding boxes
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

%% 7) CARS TURNED RED
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

%% 8) BACKGROUND BLURRING
blurredCar = imgaussfilt(carImg, 10);
carBackgroundBlurred = carImg;
carBackgroundBlurred(~repmat(BW_car, [1 1 3])) = blurredCar(~repmat(BW_car, [1 1 3]));

%% 9) SINGLE-FIGURE VISUALIZATION (12 SUBPLOTS)
figure('Name','Car Image Analysis','Position',[65 65 1000 700]);

% (1) Original Image
subplot(3,4,1);
imshow(carImg);
title('Original Image');

% (2) Top-Hat Image
subplot(3,4,2);
imshow(topHatImg, []);
title('Top-Hat Result');

% (3) Color Segmentation Mask
subplot(3,4,3);
imshow(BW_car);
title('Car Mask');

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

% (12) Original again or any extra step
subplot(3,4,12);
imshow(carImg);
title('End of Demo');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% END OF SCRIPT
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
