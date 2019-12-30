import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

def plot(ax, V, fill='green', line='red', alpha=1, with_txt=False):
	""" plots the snake onto a sub-plot
	:param ax: subplot (fig.add_subplot(abc))
	:param V: point locations ( [ (x0, y0), (x1, y1), ... (xn, yn)]
	:param fill: point color
	:param line: line color
	:param alpha: [0 .. 1]
	:param with_txt: if True plot numbers as well
	:return:
	"""
	V_plt = np.append(V.reshape(-1), V[0,:]).reshape((-1, 2))
	ax.plot(V_plt[:,0], V_plt[:,1], color=line, alpha=alpha)
	ax.scatter(V[:,0], V[:,1], color=fill,
			   edgecolors='black',
			   linewidth=2, s=50, alpha=alpha)
	if with_txt:
		for i, (x, y) in enumerate(V):
			ax.text(x, y, str(i))


def task_1():
	image = cv.imread('./data/hand.jpg')

	data = np.loadtxt('./data/hand_landmarks.txt', dtype=str, delimiter='(,)')

	kernel = np.ones((3, 3), np.float64) / 9
	image = cv.filter2D(image, -1, kernel)

	image = cv.blur(image, (3, 3))

	landmarks = np.zeros((data.shape[0], 2))
	for x in range(data.shape[0]):
		data[x] = data[x].replace('(', '')
		data[x] = data[x].replace(',', ' ')
		data[x] = data[x].replace(')', '')
		landmarks[x, :] = data[x].split()

	landmarks = np.stack((landmarks[:, 0], landmarks[:, 1]), axis=-1)

	fig = plt.figure(figsize=(10, 10))
	ax = fig.add_subplot(111)

	# ax.clear()
	# ax.imshow(image, cmap='gray')
	# # ax.set_title('frame ' + str(t))
	# plot_snake(ax, landmarks[:, 0:2])
	# plt.show()

	# 1. canny edge detection
	tr1 = 50
	tr2 = 90
	edges = cv.Canny(image, tr1, tr2)
	edges_copy = edges.copy()
	edges_copy[edges== 0]= 255
	edges_copy[edges==255]= 0
	edges = edges_copy

	# 2. distance transform

	dist = cv.distanceTransform(edges, cv.DIST_L1, 3)

	cv.normalize(dist, dist, 0, 1.0, cv.NORM_MINMAX)
	# dist = dist / np.max(dist)

	# cv.imshow('Distance Transform Image', dist)
	# cv.waitKey(0)
	# cv.destroyAllWindows()

	# 3. random init psi

	psi = np.identity(3)

	Gx = np.zeros_like(dist)
	Gy = np.zeros_like(dist)
	for i in range(dist.shape[0]):
		for j in range(dist.shape[1]):
			if (i - 1 < 0 or i + 1 >= dist.shape[0]) and (j-1 < 0 or j+1 >= dist.shape[1]):
				Gx[i, j] = -1
				Gy[i, j] = -1
				continue
			elif i-1 < 0 or i+1 >= dist.shape[0] :
				Gx[i, j]= -1
				Gy[i, j] = (dist[i, j + 1] - dist[i, j - 1]) / 2
				continue
			elif j-1<0 or j+1>=dist.shape[1]:
				Gx[i, j] = (dist[i + 1, j] - dist[i - 1, j]) / 2
				Gy[i, j] = -1
				continue

			Gx[i, j] = (dist[i + 1, j] - dist[i - 1, j]) / 2
			Gy[i, j] = (dist[i, j+1] - dist[i, j-1]) / 2

	# cv.imshow('e', Gx)
	# cv.waitKey(0)
	# cv.destroyAllWindows()
    #
	# cv.imshow('e', Gy)
	# cv.waitKey(0)
	# cv.destroyAllWindows()


	# 4. iterate:
	old_landmarks = landmarks
	for it in range (0,100):

		new_landmarks = np.zeros_like(old_landmarks)

		# 4.1. landmark_new = psi * landmark

		for x in range(old_landmarks.shape[0]):

			temp = [old_landmarks[x,0],old_landmarks[x,1],1]
			temp = np.asarray(temp).reshape(1,3)

			tt = np.matmul(temp,psi).T
			old_landmarks[x,0] = tt[0]/tt[2]
			old_landmarks[x,1] = tt[1]/tt[2]

			# 4.2. choose closest point
			w_prim = np.abs(old_landmarks[x,:])
			scnd_idx = np.abs(int(w_prim[0]))
			frst_idx = np.abs(int(w_prim[1]))

			if frst_idx>=image.shape[1] :
				frst_idx = image.shape[1]-1

			if scnd_idx>=image.shape[0] :
				scnd_idx = image.shape[0]-1


			nominator = dist[frst_idx,scnd_idx]
			denominator = np.sqrt(Gx[frst_idx,scnd_idx]**2+Gy[frst_idx,scnd_idx]**2)

			if Gx[frst_idx,scnd_idx]==0 or Gy[frst_idx,scnd_idx]==0:
				closest_point = w_prim
				new_landmarks[x, :] = np.abs(closest_point)
				continue

			if Gx[frst_idx,scnd_idx]==0 and Gy[frst_idx,scnd_idx]==0:

				continue

			if dist[frst_idx,scnd_idx]==0 :
				closest_point = w_prim
				new_landmarks[x, :] = np.abs(closest_point)
				continue


			closest_point = w_prim - ((nominator/denominator)*(Gx[frst_idx,scnd_idx]*Gy[frst_idx,scnd_idx]))

			closest_point = np.abs(closest_point).astype(int)

			new_landmarks[x, :] = closest_point



		old_landmarks = old_landmarks[new_landmarks[:, 0] != 0, :]
		new_landmarks = new_landmarks[new_landmarks[:, 0] != 0, :]


		landmarks = new_landmarks

		old_landmarks = np.asarray([old_landmarks[:, 0], old_landmarks[:, 1], np.ones((old_landmarks.shape[0]))]).T
		new_landmarks = np.asarray([new_landmarks[:, 0], new_landmarks[:, 1], np.ones((new_landmarks.shape[0]))]).T

		psi = np.matmul(np.linalg.pinv(old_landmarks), new_landmarks)

		psi[2,0]=0
		psi[2,1]=0
		psi[2,2]=1

		old_landmarks = landmarks

		ax.clear()
		ax.imshow(image, cmap='gray')
		# ax.set_title('frame ' + str(t))
		plot(ax, new_landmarks[:,0:2])
		plt.show()
		# plt.pause(0.01)




task_1()


