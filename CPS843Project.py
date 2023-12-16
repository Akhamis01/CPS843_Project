import numpy as np
from huggingface_hub import from_pretrained_keras
import keras
from PIL import Image
import cv2
import math
from skimage.metrics import structural_similarity
import time
import matplotlib.pyplot as plt

trainedModel = from_pretrained_keras("keras-io/lowlight-enhance-mirnet", compile=False)

def compressImage(pathToFile, destinationFilePath, outputImgQuality):
    originalImg = Image.open(pathToFile).convert('RGB')
    originalImgResized = originalImg.resize((256,256), Image.Resampling.LANCZOS)
    originalImgResized.save(destinationFilePath, optimize=True, quality=outputImgQuality)

    return pathToFile, destinationFilePath

pathToOriginalImage, pathToCompressedImage = compressImage(pathToFile='./testImages/testImage.png', destinationFilePath='compressedOriginalImage.jpeg', outputImgQuality=85)

def PSNR(originalImgPath, compressedImgPath):
    originalImg = cv2.imread(originalImgPath)
    originalImg = cv2.resize(originalImg, (256, 256))
    compressedImg = cv2.imread(compressedImgPath, 1) 

    mse = np.mean((originalImg - compressedImg) ** 2) 
    if(mse == 0):
        return 100
    
    max_pixel = 255.0
    psnrMetric = 20 * math.log10(max_pixel / math.sqrt(mse))

    if psnrMetric < 40:
        print('PSNR metric is less than 40 dB, please use another image.')

    return psnrMetric

PSNRValue = PSNR(pathToOriginalImage, pathToCompressedImage)
print(f"PSNR value is {PSNRValue} dB")

def prepareImageForProcessing(filePath):

    lowLightImg = Image.open(filePath).convert('RGB')
    lowLightImgResized = lowLightImg.resize((256,256),Image.NEAREST)

    imageToArray = keras.preprocessing.image.img_to_array(lowLightImgResized)
    imageToArray = imageToArray.astype('float32') / 255.0
    imageToArray = np.expand_dims(imageToArray, axis = 0)
    return imageToArray

lowLightImgArray = prepareImageForProcessing(filePath=pathToCompressedImage)

def enhanceImageUsingModel(trainedModel, lowLightImgArray):
    processedImage = trainedModel.predict(lowLightImgArray)
    outputImg = processedImage[0] * 255.0
    outputImg = outputImg.clip(0,255).reshape((np.shape(outputImg)[0],np.shape(outputImg)[1],3))
    outputImg = np.uint32(outputImg)
    displayOutputImage = Image.fromarray(outputImg.astype('uint8'),'RGB')

    return outputImg, displayOutputImage

outputImg, displayOutputImage = enhanceImageUsingModel(trainedModel, lowLightImgArray)

def prepareImageComparison(outputImg, originalImgFileName, newImgFileName):
    beforeImg = Image.open(pathToOriginalImage).convert('RGB')

    Image.fromarray(outputImg.astype('uint8'),'RGB').save(f'{newImgFileName}.png')
    afterImg = Image.open(f'./{newImgFileName}.png').convert('RGB')
    newImageWidth, newImageHeight = afterImg.size

    beforeImg = beforeImg.resize((newImageWidth, newImageHeight),Image.NEAREST).save(f'{originalImgFileName}.png')
    afterImg = afterImg.resize((newImageWidth, newImageHeight),Image.NEAREST).save(f'{newImgFileName}.png')

    return originalImgFileName, newImgFileName

originalImgFileName, newImgFileName = prepareImageComparison(outputImg, 'originalImg', 'enhancedImg')

def generateSSIMetrics(displayDiff, originalImgFileName, newImgFileName):
    before = cv2.imread(f'./{originalImgFileName}.png')
    after = cv2.imread(f'./{newImgFileName}.png')

    # Convert images to grayscale
    originalGrayScale = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
    enhancedGrayScale = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)

    (SSIMScore, diff) = structural_similarity(originalGrayScale, enhancedGrayScale, full=True)
    print("Image similarity: ", SSIMScore)

    if displayDiff:
        diff = (diff * 255).astype("uint8")
        thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]

        mask = np.zeros(before.shape, dtype='uint8')
        filled_after = after.copy()

        for c in contours:
            area = cv2.contourArea(c)
            if area > 40:
                x,y,w,h = cv2.boundingRect(c)
                cv2.rectangle(before, (x, y), (x + w, y + h), (36,255,12), 2)
                cv2.rectangle(after, (x, y), (x + w, y + h), (36,255,12), 2)
                cv2.drawContours(mask, [c], 0, (0,255,0), -1)
                cv2.drawContours(filled_after, [c], 0, (0,255,0), -1)

        cv2.startWindowThread()
        cv2.imshow('Original Image', before)
        cv2.imshow('Enhanced Image', after)
        cv2.imshow('Differences',diff)
        cv2.imshow('Mask',mask)
        cv2.imshow('Filled Enhanced Image',filled_after)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

generateSSIMetrics(displayDiff=True, originalImgFileName=originalImgFileName, newImgFileName=newImgFileName)

def plotHistogram(image, title):
    plt.hist(image.ravel(), bins=256, range=(0, 256), density=True, color='gray', alpha=0.7)
    plt.title(f'{title}')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.show()


originalImage = cv2.imread(f'./{originalImgFileName}.png')
enhancedImage = cv2.imread(f'./{newImgFileName}.png')

plotHistogram(originalImage, f'Histogram for original image')
plotHistogram(enhancedImage, f'Histogram for enhanced image')

img_rgb = cv2.cvtColor(originalImage, cv2.COLOR_BGR2RGB)
equalized_channels = [cv2.equalizeHist(channel) for channel in cv2.split(img_rgb)]
equalized_img = cv2.merge(equalized_channels)

plotHistogram(equalized_img, f'Histogram for equalized image')

img_rgb = cv2.cvtColor(originalImage, cv2.COLOR_BGR2RGB)
equalized_channels = [cv2.equalizeHist(channel) for channel in cv2.split(img_rgb)]
equalized_img = cv2.merge(equalized_channels)

plotHistogram(equalized_img, f'Histogram for equalized image')

cv2.imshow('Enhanced Image', equalized_img)
cv2.waitKey(0)
cv2.destroyAllWindows()


original = cv2.cvtColor(originalImage, cv2.COLOR_BGR2RGB)
hist, bins = np.histogram(original.flatten(), bins=256, range=[0,256])

plt.plot(hist, color='black')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.title('Luminance Histogram for Original Image')
plt.show()


enhanced = cv2.cvtColor(enhancedImage, cv2.COLOR_BGR2RGB)
hist, bins = np.histogram(enhanced.flatten(), bins=256, range=[0,256])

plt.plot(hist, color='black')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.title('Luminance Histogram for Enhanced Image')
plt.show()


