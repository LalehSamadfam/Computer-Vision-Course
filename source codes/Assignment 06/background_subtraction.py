#!/usr/bin/python3.5

import numpy as np
import cv2 as cv

'''
    load the image and foreground/background parts
    image: the original image
    background/foreground: numpy array of size (n_pixels, 3) (3 for RGB values), i.e. the data you need to train the GMM
'''


def read_image(filename):
    image = cv.imread(filename) / 255.0
    height, width = image.shape[:2]
    bounding_box = np.zeros(image.shape)
    bounding_box[90:350, 110:250, :] = 1
    bb_width, bb_height = 140, 260
    background = image[bounding_box == 0].reshape((height * width - bb_width * bb_height, 3))
    foreground = image[bounding_box == 1].reshape((bb_width * bb_height, 3))

    return image, foreground, background


class GMM(object):

    def __init__(self, mu, sigma, lambdaa):
        self.sigma = sigma
        self.mu = mu
        self.lambdaa = lambdaa

    def gaussian_scores(self, data, labels):
        return np.mean(np.divide(labels[:, 1], labels[:, 0]))

    def fit_single_gaussian(self, data, mu, sigma):
        # return (1/np.sqrt(2*np.pi)*sigma)*(np.exp((-1)*((data-mu)**2)/(2*sigma**2)))
        temp1 = (1 / np.sqrt(2 * np.pi * np.linalg.det(np.diag(sigma))))

        temp2 = np.matmul((data - mu), np.linalg.inv(np.diag(sigma)))
        # print((data-mu).T.shape)
        # print(temp2.shape)
        # print(np.matmul(temp2,(data-mu).T))
        return temp1 * (np.exp((-1 / 2) * (np.matmul(temp2, (data - mu).T))))

    def estep(self, data):

        r = np.zeros((data.shape[0], self.mu.shape[0]))

        lambdaa = self.lambdaa
        mu = self.mu
        sigma = self.sigma

        for i in range(data.shape[0]):
            # print('data number i in estep ',i)
            # for j in range (data.shape[1]):
            li = np.zeros((self.mu.shape[0], 1))
            li_sum = 0
            for k in range(self.mu.shape[0]):
                li[k] = lambdaa[k] * self.fit_single_gaussian(data[i, :], mu[k, :], sigma[k, :])
                li_sum += li[k]

            # print(r[i,:].shape)
            # print((li / li_sum).shape)

            r[i, :] = (li / li_sum).reshape(r[i, :].shape)

            # for k in range(self.mu.shape[0]):
            #     r[:,k] = li[k]/(np.sum(li))

        return r

    def mstep(self, data, r):

        lambdaa = self.lambdaa
        mu = self.mu
        sigma = self.sigma

        for k in range(self.mu.shape[0]):
            # print(r[:,k])
            # print(np.sum(r))
            lambdaa[k] = np.sum(r[:, k]) / np.sum(r)

            mu[k, :] = np.sum(np.sum(np.matmul(r[:, k], data))) / np.sum(r[:, k])

            temp = np.subtract(data, mu[k, :])
            temp1 = np.matmul(r[:, k], temp)
            sigma[k] = np.sum(np.matmul(temp1, temp.T)) \
                       / np.sum(r[:, k])

        self.lambdaa = lambdaa
        self.mu = mu
        self.sigma = sigma

        # return [lambdaa,mu,sigma]#update bayad bokonam un bala ya na?

    def em_algorithm(self, data, n_iterations=1):

        # convergance = False
        for iter in range(n_iterations):
            # expection step
            r = self.estep(data)
            # maximization step
            self.mstep(data, r)

            data = self.sample(data)
            # print('hi')
            # landa = self.landa
            # mu = self.mu
            # sigma = self.sigma
            # L = 0
            # B = 0
            # Compute Data Log Likelihood and EM Bound
            # for i in range (data.shape[0]):
            #     temp1,temp2 = 0
            #     for k in range (self.K):
            #         temp1 += landa[k]* self.fit_single_gaussian(data[i],mu[k],sigma[k])#TODO
            #         temp2 += r[i,k]*np.log(landa[k]* self.fit_single_gaussian(data[i],mu[k],sigma[k])/r[i,k])
            #
            #     L += np.log(temp1)
            #     B += temp2
            # No further improvement in L
            # if np.abs(np.subtract(B-L))<0.0001:
            #     convergance = True

    def split(self, epsilon=0.1):

        # duplicate landas
        temp_lambda = np.divide(self.lambdaa, 2)
        new_lambda = np.concatenate((temp_lambda, temp_lambda.copy()), axis=None)
        self.lambdaa = new_lambda

        # duplicate mu
        new_mu = np.zeros((self.mu.shape[0] * 2, self.mu.shape[1]))
        mu = self.mu
        sigma = self.sigma
        for k in range(mu.shape[0]):
            mu1 = mu[k, :] + (epsilon * np.var(sigma[k, :]))
            mu2 = mu[k, :] - (epsilon * np.var(sigma[k, :]))

            new_mu[k, :] = mu1
            new_mu[k + 1, :] = mu2

        self.mu = new_mu

        # duplicate covariences
        new_sigma = np.concatenate((self.sigma, self.sigma), axis=0)  # TODO
        self.sigma = new_sigma

        # fekr konam tartibe sigma ha mirize be ham. check kon

    def probability(self, data):

        label_prob = np.zeros((data.shape[0], 2))
        lambdaa = self.lambdaa
        sigma = self.sigma
        mu = self.mu

        for i in range(data.shape[0]):
            likelihood = 0
            for k in range(self.mu.shape[0]):
                print((self.fit_single_gaussian(data[i, :], mu[k, :], sigma[k, :])))
                print(lambdaa[k] * (self.fit_single_gaussian(data[i, :], mu[k, :], sigma[k, :])))

                print((self.fit_single_gaussian(data[i, :].astype(int), mu[k, :], sigma[k, :])))

                likelihood += lambdaa[k] * (self.fit_single_gaussian(data[i, :], mu[k, :], sigma[k, :]))

            w0 = likelihood * 0.8
            w1 = likelihood * 0.2

            w0 = w0 / (w0 + w1)
            w1 = w1 / (w0 + w1)

            label_prob[i, 0] = w0
            label_prob[i, 1] = w1

        return label_prob

    def sample(self, data):

        # sample h from the categorical prior
        # population = np.arange(1,self.mu.shape[0])
        # weights = self.lambdaa
        # self.lambdaa = random.choices(population, weights, self.mu.shape[0])
        # lambdaa = self.lambdaa
        np.random.shuffle(self.lambdaa)

        # self.lambdaa = np.random.shuffle(self.lambdaa)
        new_data = np.zeros_like(data)
        for k in range(self.mu.shape[0]):
            # print(np.random.multivariate_normal(self.mu[k,:], np.diag(self.sigma[k,:]), data.shape[0]).shape)
            new_data += self.lambdaa[k] * np.random.multivariate_normal(self.mu[k, :], np.diag(self.sigma[k, :]),
                                                                        data.shape[0])

        # new_data += lambdaa[k]*(self.fit_single_gaussian(data,mu[k],sigma[k,:]))
        # new_data = np.random.normal(self.mu, self.sigma, data.shape[0])

        return new_data

    def train(self, data, n_splits):

        for i in range(n_splits):
            self.split()

        self.em_algorithm(data)
        labels = self.probability(data)
        theta = self.gaussian_scores(data, labels)
        new_img = np.ones_like(data)

        new_img[labels > theta] = 0

        return new_img


image, foreground, background = read_image('person.jpg')

# cv.imshow('bg image' , background)
# cv.waitKey(0)
# cv.destroyAllWindows()

'''
TODO: compute p(x|w=background) for each image pixel and manipulate the image such that everything below the threshold is black, display the resulting image
Hint: Slide 64
'''

lambdaa = 1

# mu = np.zeros((1,3))
mu = np.mean(background, axis=0).reshape(1, 3)
sigma = np.var(background, axis=0).reshape((1, 3))

gmm_background = GMM(mu, sigma, lambdaa)

# gmm_background.init(mu, sigma, lambdaa)
new_image = gmm_background.train(background, 3)

cv.imshow('new image', new_image)
cv.waitKey(0)
cv.destroyAllWindows()