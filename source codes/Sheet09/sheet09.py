import numpy as np
import os
import time
import cv2 as cv
import matplotlib.pyplot as plt


def load_FLO_file(filename):
    assert os.path.isfile(filename), 'file does not exist: ' + filename   
    flo_file = open(filename,'rb')
    magic = np.fromfile(flo_file, np.float32, count=1)
    assert magic == 202021.25,  'Magic number incorrect. .flo file is invalid'
    w = np.fromfile(flo_file, np.int32, count=1)
    h = np.fromfile(flo_file, np.int32, count=1)
    data = np.fromfile(flo_file, np.float32, count=2*w[0]*h[0])
    flow = np.resize(data, (int(h[0]), int(w[0]), 2))
    flo_file.close()
    return flow

class OpticalFlow:
    def __init__(self):
        # Parameters for Lucas_Kanade_flow()
        self.EIGEN_THRESHOLD = 0.01 # use as threshold for determining if the optical flow is valid when performing Lucas-Kanade
        self.WINDOW_SIZE = [25, 25] # the number of points taken in the neighborhood of each pixel

        # Parameters for Horn_Schunck_flow()
        self.EPSILON= 0.002 # the stopping criterion for the difference when performing the Horn-Schuck algorithm
        self.MAX_ITERS = 1000 # maximum number of iterations allowed until convergence of the Horn-Schuck algorithm
        self.ALPHA = 1.0 # smoothness term

        # Parameter for flow_map_to_bgr()
        self.UNKNOWN_FLOW_THRESH = 1000

        self.prev = None
        self.next = None

    def next_frame(self, img):
        self.prev = self.next
        self.next = img

        if self.prev is None:
            return False

        frames = np.float32(np.array([self.prev, self.next]))
        frames /= 255.0

        #calculate image gradient
        self.Ix = cv.Sobel(frames[0], cv.CV_32F, 1, 0, 3)
        self.Iy = cv.Sobel(frames[0], cv.CV_32F, 0, 1, 3)
        self.It = frames[1]-frames[0]

        return True

    #***********************************************************************************
    # function for converting flow map to to BGR image for visualisation
    # return bgr image
    def flow_map_to_bgr(self, flow):
        flow_bgr = None



        hsv = np.zeros((flow.shape[0]* flow.shape[1],3))
        hsv[..., 1] = 255
        # yasy
        # print(flow[...,0].flatten().shape)
        # print(flow[...,1].flatten().shape)

        mag, ang = cv.cartToPolar(flow[..., 0].flatten(), flow[..., 1].flatten())
        hsv[..., 0] = (ang * 180 / np.pi / 2).reshape(ang.shape[0])
        hsv[..., 2] = (cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)).reshape(ang.shape[0])
        hsv = hsv.reshape(flow.shape[0], flow.shape[1],3).astype(np.float32)

        # cv.imshow('hsv', hsv)
        # cv.waitKey(0)
        # cv.destroyAllWindows()

        # inja tabdil be bgr ke mishe ye etefaghi miofte meshkish mikone

        flow_bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)


        return flow_bgr

    #***********************************************************************************
    def compute_uv(self, Ix, Iy, It):
        # in order to make matrix I
        first = np.sum(np.matmul( Ix , Ix))
        second = np.sum(np.matmul(Ix , Iy))
        third = np.sum(np.matmul(Ix , Iy))
        fourth = np.sum(np.matmul(Iy , Iy))

        ATA = np.zeros((2, 2))
        ATA[0, 0] = first
        ATA[0, 1] = second
        ATA[1, 0] = third
        ATA[1, 1] = fourth

        b1 = np.sum(np.matmul(Ix, It))
        b2 = np.sum(np.matmul(Iy, It))

        ATb = np.zeros((2, 1))
        ATb[0] = -b1
        ATb[1] = -b2

        uv = np.matmul(np.linalg.inv(ATA),ATb)

        return uv
    # implement Lucas-Kanade Optical Flow 
    # returns the Optical flow based on the Lucas-Kanade algorithm and visualisation result
    def Lucas_Kanade_flow(self):
        flow = None


        current_frame = self.next

        flow = np.zeros((current_frame.shape[0], current_frame.shape[1], 2))
        u = np.zeros((current_frame.shape[0], current_frame.shape[1]))
        v = np.zeros((current_frame.shape[0], current_frame.shape[1]))
        half_window_size = int(self.WINDOW_SIZE[0] / 2)
        # for i in range(half_window_size, current_frame.shape[0]-half_window_size):
        #     for j in range(half_window_size, current_frame.shape[1]-half_window_size):
        #
        #         ind1 = i - half_window_size
        #         ind2 = i + half_window_size
        #         ind3 = j - half_window_size
        #         ind4 = j + half_window_size
        #
        #         if i - half_window_size < 0: ind1 = 0
        #         if j - half_window_size < 0: ind3 = 0
        #         if i + half_window_size >= current_frame.shape[0]: ind2 = current_frame.shape[0] - 1
        #         if j + half_window_size >= current_frame.shape[1]: ind2 = current_frame.shape[1] - 1
        #
        #         window_Ix = self.Ix[ind1:ind2 + 1, ind3:ind4 + 1]
        #         window_Iy = self.Iy[ind1:ind2 + 1, ind3:ind4 + 1]
        #
        #         window_It = self.It[ind1:ind2 + 1, ind3:ind4 + 1]
        #
        #         uv = self.compute_uv(window_Ix, window_Iy, window_It)
        #
        #         u[i,j] = uv[0]
        #         v[i,j] = uv[1]
        #         if abs(u[i, j]) > self.EIGEN_THRESHOLD or abs(v[i, j]) > self.EIGEN_THRESHOLD:
        #             flow[i,j,:] = uv.reshape(2,)

        # calculate the gradient product accumulations for each pixel
        Sxx = cv.boxFilter(self.Ix ** 2, -1, ksize=(half_window_size,) * 2, normalize=True)
        Sxy = cv.boxFilter(self.Ix * self.Iy, -1, ksize=(half_window_size,) * 2, normalize=True)
        Syy = cv.boxFilter(self.Iy ** 2, -1, ksize=(half_window_size,) * 2, normalize=True)
        Sxt = cv.boxFilter(self.Ix * self.It, -1, ksize=(half_window_size,) * 2, normalize=True)
        Syt = cv.boxFilter(self.Iy * self.It, -1, ksize=(half_window_size,) * 2, normalize=True)
        # del It, Ix, Iy

        # calculate the displacement matrices U, V
        rows, cols = current_frame.shape
        flow = np.zeros((rows, cols, 2), dtype=np.float32)
        A = np.dstack((Sxx, Sxy, Sxy, Syy))
        b = np.dstack((-Sxt, -Syt))
        for r in range(rows):
            for c in range(cols):
                flow[r, c, :] = np.linalg.lstsq(A[r, c].reshape((2, 2)), b[r, c])[0]

        # cv.imshow('u', u)
        # cv.waitKey(0)
        # cv.destroyAllWindows()
        #
        # cv.imshow('v', v)
        # cv.waitKey(0)
        # cv.destroyAllWindows()


        flow_bgr = self.flow_map_to_bgr(flow)
        return flow, flow_bgr

    #***********************************************************************************
    # implement Horn-Schunck Optical Flow 
    # returns the Optical flow based on the Horn-Schunck algorithm and visualisation result
    def Horn_Schunck_flow(self):
        flow = None

        current_frame = self.next

        flow = np.zeros((current_frame.shape[0], current_frame.shape[1], 2))
        u = np.zeros((current_frame.shape[0], current_frame.shape[1]))
        v = np.zeros((current_frame.shape[0], current_frame.shape[1]))
        half_window_size = int(self.WINDOW_SIZE[0] / 2)
        convergance = False
        k = np.array([[0, 1/4, 0],
                   [1/4,    -1, 1/4],
                   [0, 1/4, 0]])
        alpha = 1

        while(not convergance):

            prev_u = u
            prev_v = v

            delta_u = cv.filter2D(u, -1, k)
            delta_v = cv.filter2D(v, -1, k)

            u_bar = u + delta_u
            v_bar = v + delta_v

            der = (self.Ix*u_bar + self.Iy*v_bar + self.It)/(alpha**2 + self.Ix**2 + self.Iy**2)

            u = u_bar - (self.Ix * der)
            v = v_bar - (self.Iy * der)

            diff = np.sum(np.abs(u - prev_u) + np.abs(v - prev_v))
            print('diff: ', diff)
            if(diff < 0.002):
                convergance = True

        flow[..., 0] = u
        flow[..., 1] = v

        cv.imshow('u', u)
        cv.waitKey(0)
        cv.destroyAllWindows()

        cv.imshow('v', v)
        cv.waitKey(0)
        cv.destroyAllWindows()

        flow_bgr = self.flow_map_to_bgr(flow)
        return flow, flow_bgr

    #***********************************************************************************
    #calculate the angular error here
    # return average angular error and per point error map
    def calculate_angular_error(self, estimated_flow, groundtruth_flow):
        aae = None
        aae_per_point = None

        uc= groundtruth_flow[..., 0].flatten()
        vc= groundtruth_flow[..., 1].flatten()

        u = estimated_flow[..., 0].flatten()
        v = estimated_flow[..., 1].flatten()

        n = u.shape[0]
        sum = 0
        aae_per_point = np.zeros((n,1))
        for i in range (n):
            nominator = uc[i]*u[i] + vc[i]*v[i] + 1
            denominator = np.sqrt((uc[i]**2 + vc[i]**2 + 1) + (u[i]**2 + v[i]**2 + 1))
            # sum += np.arccos(nominator/denominator)
            # if (np.isnan(sum)):
            #     break
            aae_per_point[i] = np.arccos(nominator/denominator)

        aae = np.sum(aae_per_point, dtype=np.int64)/n
        aae_per_point = aae_per_point.reshape(estimated_flow.shape[0] , estimated_flow.shape[1])
        return aae, aae_per_point


if __name__ == "__main__":

    data_list = [
        'data/frame_0001.png',
        'data/frame_0002.png',
        'data/frame_0007.png',
    ]

    gt_list = [
        './data/frame_0001.flo',
        './data/frame_0002.flo',
        './data/frame_0007.flo',
    ]

    Op = OpticalFlow()
    
    for (i, (frame_filename, gt_filemane)) in enumerate(zip(data_list, gt_list)):
        groundtruth_flow = load_FLO_file(gt_filemane)
        img = cv.cvtColor(cv.imread(frame_filename), cv.COLOR_BGR2GRAY)
        if not Op.next_frame(img):
            continue

        flow_lucas_kanade, flow_lucas_kanade_bgr = Op.Lucas_Kanade_flow()
        aae_lucas_kanade, aae_lucas_kanade_per_point = Op.calculate_angular_error(flow_lucas_kanade, groundtruth_flow)
        print('Average Angular error for Luacas-Kanade Optical Flow: %.4f' %(aae_lucas_kanade))

        flow_horn_schunck, flow_horn_schunck_bgr = Op.Horn_Schunck_flow()
        aae_horn_schunk, aae_horn_schunk_per_point = Op.calculate_angular_error(flow_horn_schunck, groundtruth_flow)
        print('Average Angular error for Horn-Schunck Optical Flow: %.4f' %(aae_horn_schunk))

        flow_bgr_gt = Op.flow_map_to_bgr(groundtruth_flow)

        fig = plt.figure(figsize=(img.shape))

        # Display
        fig.add_subplot(2, 3, 1)
        plt.imshow(flow_bgr_gt)
        fig.add_subplot(2, 3, 2)
        plt.imshow(flow_lucas_kanade_bgr)
        fig.add_subplot(2, 3, 3)
        plt.imshow(aae_lucas_kanade_per_point)
        fig.add_subplot(2, 3, 4)
        plt.imshow(flow_bgr_gt)
        fig.add_subplot(2, 3, 5)
        plt.imshow(flow_horn_schunck_bgr)
        fig.add_subplot(2, 3, 6)
        plt.imshow(aae_horn_schunk_per_point)
        plt.show()

        print("*"*20)
