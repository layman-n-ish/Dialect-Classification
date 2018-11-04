# Dialect-Classification

Given the title, it doesn't take a Sherlock Holmes to decipher the problem statement; classify the dialects of British English, spoken across nine different regions of the British Isles. 

## Data? 
The IViE (Intonational Variation in English) speech corpus is used for this particular project. Check out their website for more information on how the data was collected, what the data composes of, etc. 

## Is solving this problem worth the effort? 
Dialects are one of the most important factors that influence speech system performance including ones like Automatic Speech Recognition (ASR). Building a high quality speech recognizer consisting of an ensemble of dialect-specific recognizers could be a plausible solution. 

Information of a person's origin can also be extracted using this technology. Does this qualify as a personal data breach?

## Feature Extraction
Statistical features (mean, variance, etc.) of a number of attributes like MFCC (Mel Frequency Cepstrum Coefficients), Filterbank coefficients, delta-delta coefficients, chroma, RMSE, etc. are computed frame-wise (frame length of around 23 ms) for every audio sample in a given class. Why these features? Check out the **'Materials'** file for a bunch of links to lay the theoretical foundation.


