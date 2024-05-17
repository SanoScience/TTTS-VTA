# Visual Tracking Algorithm
This repository is a implementation of visual tracking algorithm (VTA) proposed in paper [Vision Based Robot Assistance in TTTS Fetal Surgery](https://ieeexplore.ieee.org/iel7/8844528/8856280/08856402.pdf). This algorithm can be used to do segmentation of fetoscopic images for the pourpose of mosaicing.

Data used for testing the algorithm comes form [FetReg](https://www.ucl.ac.uk/interventional-surgical-sciences/weiss-open-research/weiss-open-data-server/fetreg-largescale-multi-centre-fetoscopy-placenta-dataset) dataset first introduced in 2021.

## Algorithm description
Algorithm can be divided into 5 steps. By following them we should get expected results presented in the paper.

- ### Extract green channel
    Firstly algorithm extracts green channel from RGB images because this channel has higher contrast between the vascular structures and the background according to [this](https://ieeexplore.ieee.org/document/6566372) and [this](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/7962.toc#_=_) papers.
- ### Apply median blur filter
    Than to remove noise median blur filter is applied.
- ### Compute CLAHE
    Next, to enhance the contrast between the different structures in the image the algorithm computes a Contrast Limited Adaptive Histogram Equalization (CLAHE).
- ### Apply thresholding binarization
    After that an adaptive thresholding binarization is applied using the algorithm in [here](https://www.researchgate.net/publication/220494200_Adaptive_Thresholding_using_the_Integral_Image)
- ### Postprocessing filters
    Lastly algorithm applies two postprocessing filters:
    - Specle filter - to eliminate the small componentsof the binary image
    - Morphological closing - filter to fill the small holes in the detected structures

## Requirements
- OpenCV

## Todo
- create loop of mosaics
- refactor code
- test 
- finish readme