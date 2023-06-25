import numpy as np
import os
import math
import glob
import cv2
import matplotlib.pyplot as plt
import scipy.optimize as optimize
import copy
import scipy
from scipy import linalg
from scipy.stats import norm

# Read  all images

def read_images():
    folder = os.path.join("Calibration_Imgs")
    if os.path.exists(folder): 
        list = os.listdir(folder)
        list.sort()
    else:
        raise Exception ("No such directory")
    images_path = []
    for i in range(len(list)):
        image_path = os.path.join(folder,list[i])
        images_path.append(image_path)
    images = []
    for n in (images_path): 
        img = cv2.imread(n)
        images.append(img)
    return images


def world(l,b,sq_size):        #9,6,21.5
  world_co_x, world_co_y = np.meshgrid(range(l), range(b))
  X = world_co_x.reshape((l*b),1)
  Y = world_co_y.reshape((l*b),1)
  world_co = np.array(np.hstack((X, Y))*sq_size)
  world_co = np.float32(world_co)
  return world_co


def homography(images,worldpt):
    temp = copy.deepcopy(images)
    H_mtx = []
    camera_co = []
    for i, img in enumerate(temp):
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray_img, (9, 6), None)
        # print(ret)
        if ret == True:
            corners = np.reshape(corners,(-1,2))
            H, mask = cv2.findHomography(worldpt, corners, cv2.RANSAC, 5.0)     
            H_mtx.append(H)
            camera_co.append(corners)
            draw =  cv2.drawChessboardCorners(img, (9, 6), corners, True)
            cv2.imshow('img' ,draw)
            cv2.waitKey(0)
    return H_mtx, camera_co


def v(H, i, j):
  H=H.T
  v_ij=np.array([
        H[i][0] * H[j][0],
        H[i][0] * H[j][1] + H[i][1] * H[j][0],
        H[i][1] * H[j][1],
        H[i][2] * H[j][0] + H[i][0] * H[j][2],
        H[i][2] * H[j][1] + H[i][1] * H[j][2],
        H[i][2] * H[j][2]])
  return v_ij.T


def B_mtx(H_mtx):
  V = []
  for i in range (len(H_mtx)):
      H = H_mtx[i]
      v12 = (v(H, 0, 1)).T
      v11_v22 = (v(H, 0, 0).T- v(H, 1, 1).T)
      V.append(v12)
      V.append(v11_v22)
  V = np.array(V)

  _, _, Vh = np.linalg.svd(V, full_matrices=True)      #eigen vector of v.Tv associated with smallest eigen value      
  b = Vh.T[:,-1]
  return b


def A_mtx(b):
  v0 = (b[1]*b[3] - b[0]*b[4])/(b[0]*b[2] - b[1]**2)
  lmbd = b[5] - (b[3]**2 + v0*(b[1]*b[3] - b[0]*b[4]))/b[0]
  alpha = np.sqrt(lmbd/b[0])
  beta = np.sqrt(lmbd*b[0] / (b[0]*b[2] - b[1]**2))
  gamma = -1*b[1]*(alpha**2)*beta/lmbd
  u0 = (gamma*v0/beta) - (b[3]*(alpha**2)/lmbd)
  print("lmbd",lmbd)


  intrinsic = np.array([[alpha, gamma, u0],
                [0, beta, v0],
                [0, 0, 1]])
  return intrinsic


def Rt_mtx(intrinsic, H_mtx):
    ex = []
    for h in H_mtx:

      h1 = h[:, 0]
      h2 = h[:, 1]
      h3 = h[:, 2]

      den = scipy.linalg.norm(np.dot(np.linalg.inv(intrinsic),h1), ord =2)
      lamda =  1/ den
      r1 = lamda * np.dot(np.linalg.inv(intrinsic), h1) 
      r2 = lamda * np.dot(np.linalg.inv(intrinsic), h2) ## No r3 as z=0 
      t = lamda * np.dot(np.linalg.inv(intrinsic), h3)

      tran_mtx = np.transpose(np.vstack((r1, r2, t)))
      ex.append(tran_mtx)
    return ex


def calib_matrix(parameter):
    alpha, gamma, beta, u0, v0, K1, K2 = parameter
    A = np.array([[alpha, gamma, u0],
                  [0, beta, v0],
                  [0, 0, 1]])
    distortion = np.array([[K1],[K2]])
    return A, distortion


def find_parameters(K, distortion):
    return np.array([K[0,0], K[0,1], K[1,1], K[0,2], K[1,2], distortion[0], distortion[1]])


def fun(x0, extrinsic, camera_co, worldpt):
    K, k_distortion = calib_matrix(x0)
    total_error, _ = projection_error(K, k_distortion, extrinsic, camera_co, worldpt)
    return np.array(total_error)


# Minimizing the grometric error
def projection_error(K, k_distortion, extrinsic, camera_co, worldpt):
    alpha, gamma, beta, u0, v0, k1, k2 = find_parameters(K, k_distortion)
    total_error = []
    proj_pt  =[]
    for i in range(len(camera_co)):
        i_corner = camera_co[i]
        A_Rt_mtx = np.dot(K, extrinsic[i])
        err = 0
        reproj_pt = []
        for j in range(len(i_corner)):
            w_pt = worldpt[j]
            M = np.array([[w_pt[0]], [w_pt[1]], [1]])

            m_ij = i_corner[j]
            m_ij = np.array([[m_ij[0]],[m_ij[1]],[1]], dtype = float)
  
            projected_pt = np.matmul(extrinsic[i], M)
            projected_pt = projected_pt/ projected_pt[2]
            x =  projected_pt[0]
            y = projected_pt[1]

            img_co = np.matmul(A_Rt_mtx , M)
            img_co=img_co/img_co[2]
            u =  img_co[0]
            v = img_co[1] 


            r = (np.square(x)+ np.square(y))
            u_hat = u + (u - u0)*(k1*r+k2*(np.square(r)))
            v_hat = v + (v - v0)*(k1*r+k2*(np.square(r)))

            m_ij_hat = np.array([[u_hat], [v_hat], [1]], dtype = float)
            reproj_pt.append((m_ij_hat[0],m_ij_hat[1]))
            error = np.linalg.norm((m_ij - m_ij_hat), ord=2)
            err = err + error

        proj_pt.append(reproj_pt)
        total_error.append(err / 54)
    return np.array(total_error), np.array(proj_pt)


def plot(img, pt):
  img = copy.deepcopy(img)
  for i in range(len(pt)):
    draw = cv2.circle(img, (int(pt[i][0]),int(pt[i][1])), 7, (255,0,0), -2)
  cv2.imshow('img' ,draw)
  cv2.waitKey(0)


def main():
    images = read_images()
    worldpt = world(9,6,21.5)
    homo,camera_co=homography(images,worldpt)
    bvector = B_mtx(homo)
    intrinsic = A_mtx(bvector)
    print("intrinsic",intrinsic)
    extrinsic = Rt_mtx(intrinsic,homo)
    k_distortion = np.array([[0],[0]])
    parameter = find_parameters(intrinsic, k_distortion)
    optimization = optimize.least_squares(fun, x0=parameter, method="lm", args=[extrinsic, camera_co, worldpt])
    res = optimization.x
    A_opt, K_opt = calib_matrix(res)
    print("A_opt",A_opt)
    print("K_opt",K_opt)
    total_error_bef,_ = projection_error(intrinsic, k_distortion, extrinsic, camera_co, worldpt)
    print("total_error_before:", total_error_bef)
    total_error, reprojected_pt = projection_error(A_opt, K_opt, extrinsic, camera_co, worldpt)
    print("total_error:", total_error)
    mean_error = np.mean(total_error)
    K_opt = np.array([K_opt[0], K_opt[1], 0, 0, 0], dtype = float)
    print("Reprojection error", mean_error)
    for i in range(len(images)):
        temp_img = images[i]
        temp_img = cv2.undistort(temp_img, A_opt, K_opt)
        plot(temp_img, reprojected_pt[i])


if __name__ == "__main__":
    main()