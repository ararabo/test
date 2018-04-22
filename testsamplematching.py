# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 16:37:17 2018

@author: ararabo
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt

strSampleImgPath='./sampleimage/'
strImageIcon='image/'
strImgFileName='image_'
strImgFileNameQry='select_001+'
strExt_png='.png'
strExt_jpg='.jpg'
k=59
vLength=np.zeros(6)
vChoice=np.zeros(6)
strImgFile=strSampleImgPath+strImageIcon+strImgFileName+str(k)+strExt_png
img1=cv2.imread(strImgFile,0)        # queryImage
i=0
for j in range(6):
    strImgFileQry=strSampleImgPath+strImgFileNameQry+str(j)+strExt_jpg
    img2 = cv2.imread(strImgFileQry,0)
    
    # Initiate ORB detector
    orb = cv2.ORB_create()
    
    ## find the keypoints and descriptors with ORB
    #kp1, des1 = orb.detectAndCompute(img1,None)
    #kp2, des2 = orb.detectAndCompute(img2,None)
    
    # create BFMatcher object
    # SIFT特徴検出器を始める
    sift = cv2.xfeatures2d.SIFT_create()
    
    # SIFTを用いてキーポイントと特徴記述子を求める
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    
    # デフォルトのパラメタを使ったBFMatcher
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2, k=2)
    
    # 割合試験を適用
    good = []
    for m,n in matches:
        if m.distance < 0.70*n.distance:
            good.append([m])
    sLenthGood=len(good)
    vLength[j]=sLenthGood
    vChoice[i]=max(vLength)
    # cv2.drawMatchesKnn はmatchesとしてリストを期待
    img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good, None,flags=2)
    
    plt.imshow(img3),plt.show()