import numpy as np
import matplotlib.pylab as plt

observations = np.load('observations.npy')


def get_observation(t):
    return observations[t]


class KalmanFilter(object):
    def __init__(self, psi, sigma_p, phi, sigma_m, tau):
        self.psi = psi
        self.sigma_p = sigma_p
        self.phi = phi
        self.sigma_m = sigma_m
        self.state = None
        self.covariance = None
        self.tau = tau
        self.c = 1  # may need to be changed

    def init(self, init_state):
        self.state = np.zeros((self.tau, 4))
        self.state = init_state
        self.covariance = np.array([[self.c, 0, 0, 0],
                                    [0, self.c, 0, 0],
                                    [0, 0, self.c, 0],
                                    [0, 0, 0, self.c]])
    def track(self, xt):
        # state_prediction
        state_trans_mat = np.zeros((self.tau + 1 , self.tau + 1, 4, 4))
        state_trans_mat[0][0] = self.psi
        for i in range(self.tau - 1):
            state_trans_mat[i + 1][i] = np.eye(4)
        state_predicted = np.dot(self.psi, self.state)

        # cov_prediction
        cov_predicted = self.sigma_p + np.dot(np.dot(self.psi, self.covariance), np.transpose(self.psi))

        # Kalman gain
        inner = np.dot(self.phi, np.dot(cov_predicted, np.transpose(self.phi)))
        right = np.linalg.inv(self.sigma_m + inner)
        kalman_gain = np.dot(np.dot(cov_predicted, np.transpose(self.phi)), right)

        # state_update
        self.state = state_predicted + np.dot(kalman_gain, xt - np.dot(self.phi, state_predicted))

        # cov_update
        right = np.dot(kalman_gain, self.phi)
        self.convariance = np.dot(np.eye(4) - right, cov_predicted)

    def get_current_location(self):
        return [self.state[0], self.state[1]]


def perform_tracking(tracker):
    track = []
    for t in range(len(observations)):
        tracker.track(get_observation(t))
        track.append(tracker.get_current_location())
    return track


def main():
    init_state = np.array([1, 0, 0, 0])

    psi = np.array([[1, 0, 1, 0],
                    [0, 1, 0, 1],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])
    sp = 0.01
    sigma_p = np.array([[sp, 0, 0, 0],
                        [0, sp, 0, 0],
                        [0, 0, sp * 4, 0],
                        [0, 0, 0, sp * 4]])

    phi = np.array([[1, 0, 0, 0],
                    [0, 1, 0, 0]])
    sm = 0.05
    sigma_m = np.array([[sm, 0],
                        [0, sm]])

    tracker = KalmanFilter(psi, sigma_p, phi, sigma_m, tau=0)
    tracker.init(init_state)

    fixed_lag_smoother = KalmanFilter(psi, sigma_p, phi, sigma_m, tau=5)
    fixed_lag_smoother.init(init_state)

    track = perform_tracking(tracker)
    track_smoothed = perform_tracking(fixed_lag_smoother)

    plt.figure()
    plt.plot([x[0] for x in observations], [x[1] for x in observations])
    plt.plot([x[0] for x in track], [x[1] for x in track])
    plt.plot([x[0] for x in track_smoothed], [x[1] for x in track_smoothed])

    plt.show()


if __name__ == "__main__":
    main()
