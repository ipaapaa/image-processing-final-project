%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Image Processing Final Project
% Submitted by: Marc Olata and Job Isaac Ong (TN36)
%
% This script processes two images:
%   1) "birdies.jpg" (object of interest: birds) 
%      - Birds are turned red.
%   2) "doggo.jpg" (object of interest: dog)
%      - The dog is turned yellow.
%
% For each image, the following outputs are produced:
%   (1) Original Image
%   (2) HSV Image
%   (3) YCbCr Image
%   (4) Sobel Edges
%   (5) Prewitt Edges
%   (6) Enhanced (Dilated) Edges
%   (7) (Refined) Color Segmentation Mask
%   (8) Segmented Overlay
%   (9) K-means Segmentation
%   (10) Object Detection Overlay (with thicker bounding boxes)
%   (11) Final Color Overlay (birds turned red / dog turned yellow with partial opacity)
%   (12) Background Blurring Output
%
% Additional revisions:
%   - Bounding boxes drawn with a thicker line.
%   - Prewitt edge detection is added.
%   - Background blurring is added.
%
% Two separate figures are created: one for "birdies.jpg" and one for "doggo.jpg".
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% ------------------- BIRD IMAGE ANALYSIS -------------------

% 1) READ & RESIZE IMAGE
originalImg = imread('birdies.jpg');
birdImg = imresize(originalImg, [512 512]);
grayBird = rgb2gray(birdImg);

% 2) COLOR SPACE CONVERSIONS
hsvBird   = rgb2hsv(birdImg);
ycbcrBird = rgb2ycbcr(birdImg);

% 3) COLOR SEGMENTATION (RGB Example)
% Threshold for "dark" pixels to capture bird silhouettes.
R = birdImg(:,:,1);
G = birdImg(:,:,2);
B = birdImg(:,:,3);
darkThreshold = 70;
BW_bird = (R < darkThreshold) & (G < darkThreshold) & (B < darkThreshold);
BW_bird = bwareaopen(BW_bird, 50);

% Create overlay: only segmented (bird) regions remain.
colorSegmentedOverlayBird = birdImg;
colorSegmentedOverlayBird(repmat(~BW_bird, [1 1 3])) = 0;

% 4) EDGE DETECTION
BW_sobel_bird   = edge(grayBird, 'sobel');
BW_prewitt_bird = edge(grayBird, 'prewitt');

% 5) EDGE ENHANCEMENT (Dilation)
se = strel('disk', 1);
BW_sobel_dilated_bird = imdilate(BW_sobel_bird, se);

% 6) K-MEANS CLUSTERING
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

% 7) OBJECT DETECTION & THICKER BOUNDING BOXES
CC = bwconncomp(BW_bird);
stats = regionprops(CC, 'BoundingBox', 'Centroid');
% Use insertShape to draw thicker boxes:
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

% 8) TURN THE BIRDS RED (Full replacement in mask)
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

% 9) BACKGROUND BLURRING: Blur background while keeping the object sharp
blurredBird = imgaussfilt(birdImg, 10);
birdBackgroundBlurred = birdImg;
birdBackgroundBlurred(~repmat(BW_bird, [1 1 3])) = blurredBird(~repmat(BW_bird, [1 1 3]));

% 10) DISPLAY BIRD ANALYSIS (12 Subplots)
figure('Name', 'Bird Image Analysis', 'Position', [50 50 1400 900]);
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

%% ------------------- DOG IMAGE ANALYSIS -------------------

% 1) READ & RESIZE IMAGE
originalDog = imread('dog.jpg');
dogImg = imresize(originalDog, [512 512]);
grayDog = rgb2gray(dogImg);

% 2) COLOR SPACE CONVERSIONS
hsvDog   = rgb2hsv(dogImg);
ycbcrDog = rgb2ycbcr(dogImg);

% 3) REFINED COLOR SEGMENTATION
% Exclude strongly green areas (grass)
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

% 4) EDGE DETECTION
BW_sobel_dog   = edge(grayDog, 'sobel');
BW_prewitt_dog = edge(grayDog, 'prewitt');

% 5) EDGE ENHANCEMENT (Dilation)
se = strel('disk',1);
BW_sobel_dog_dilated = imdilate(BW_sobel_dog, se);

% 6) K-MEANS CLUSTERING
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

% 7) OBJECT DETECTION & THICKER BOUNDING BOXES (Dog)
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

% 8) CHANGE DOG'S COLOR TO YELLOW (with partial opacity)
alpha = 0.7;
dogYellowOverlay = im2double(dogImg);
yellowOverlay = ones(size(dogImg));
yellowOverlay(:,:,1) = 1;
yellowOverlay(:,:,2) = 1;
yellowOverlay(:,:,3) = 0;
mask = repmat(BW_dog, [1 1 3]);
dogYellowOverlay(mask) = alpha * yellowOverlay(mask) + (1 - alpha) * dogYellowOverlay(mask);
dogYellowOverlay = im2uint8(dogYellowOverlay);

% 9) BACKGROUND BLURRING (Dog)
blurredDog = imgaussfilt(dogImg, 10);
dogBackgroundBlurred = dogImg;
dogBackgroundBlurred(~repmat(BW_dog, [1 1 3])) = blurredDog(~repmat(BW_dog, [1 1 3]));

% 10) DISPLAY DOG ANALYSIS (12 Subplots)
figure('Name', 'Dog Image Analysis', 'Position', [100 100 1400 900]);
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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% END OF SCRIPT
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

