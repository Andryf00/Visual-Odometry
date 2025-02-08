from flow_net.flow_utils import Network, estimate
from depth_net.depth_net import Depth_net
import numpy as np
import PIL.Image
from torchvision import transforms
from sklearn import linear_model

import torch
import cv2 as cv

#FLOW_THRESHOLD = 10

#use Ransac to estimate scaling factor
def estimate_scaling_factor(d, d_prime):
    ransac = linear_model.RANSACRegressor(
                        estimator=linear_model.LinearRegression(
                            fit_intercept=False),
                        min_samples=15,
                        max_trials=100,
                        stop_probability=0.99,
                        residual_threshold=0.1,
                        )
    
    d_mask = d>0
    d_prime_mask = d_prime>0
    count = 0
    mask = d_mask*d_prime_mask
    d_non_zero = d*mask
    d_prime_non_zero = d_prime*mask
    ransac.fit(
        d_prime_non_zero.reshape(-1, 1),
        d_non_zero.reshape(-1, 1),
                )
    scale = ransac.estimator_.coef_[0, 0]
    return scale

def triangulate(kp1, kp2, R, t, K):
    eye = np.eye(3)
    zeros = np.zeros(t.shape).reshape(3, 1)

    P1_proj = np.matmul(K, np.concatenate((eye, zeros), axis=1))
    P2_proj = np.matmul(K, np.concatenate((R, t.reshape(3, 1)), axis=-1))
    
    kp1_norm = kp1.copy()
    kp2_norm = kp2.copy()
    kp1_norm[:, 0] = \
        (kp1[:, 0] - K[2,0]) /  K[0,0]
    kp1_norm[:, 1] = \
        (kp1[:, 1] - K[2,1]) / K[1,1]
    kp2_norm[:, 0] = \
        (kp2[:, 0] - K[2,0]) / K[0,0]
    kp2_norm[:, 1] = \
        (kp2[:, 1] - K[2,1]) / K[1,1]
    triangulated_points = cv.triangulatePoints(P1_proj[:3], P2_proj[:3], kp1_norm.astype(np.float64), kp2_norm.astype(np.float64))
    triangulated_points = triangulated_points.astype(np.float64)
    print("triang_done")
    triangulated_points /= triangulated_points[3]
    X2 = P2_proj[:3] @ triangulated_points
    
    #points = cv.convertPointsFromHomogeneous(triangulated_points)
    print("DONE")
    #print(X2[2])
    #points_euclidean = triangulate_points(P1_proj.detach().numpy(), P2_proj.detach().numpy(), points, matched_points)
    return X2[2]

def simple_scale_recovery(kp1, kp2, R, t, K, depth):
    
    d_prime = triangulate(kp1, kp2, R, t, K)
    return estimate_scaling_factor(depth, d_prime)
