import argparse
import cPickle
import os
import platform
import random
import requests

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorlayer as tl
from PIL import Image


os.environ["CUDA_VISIBLE_DEVICES"] = "2"


def getArgs():
	parser = argparse.ArgumentParser()
	parser.add_argument("-m", "--mode", help="choose train or eval",
	                    type=str, default="train", choices=["train", "eval"])
	parser.add_argument("-l", "--learnRate",
	                    help="input the learn rate", type=float, default=1.)
	parser.add_argument("-b", "--batchSize",
	                    help="input the batch size", type=int, default=125)
	parser.add_argument("-p", "--checkpointPath",
	                    help="input the path of checkpoint", type=str, default="ckp")
	parser.add_argument(
		"-t", "--testStep", help="the step of test", type=int, default=10)
	parser.add_argument(
		"-s", "--saveStep", help="the step of save checkpoint", type=int, default=500)
	args = parser.parse_args()
	return args


class cnn:
	def __init__(self, learnRate, batchSize, checkpointPath, testStep, saveStep):
		self.learnRate = learnRate
		self.batchSize = batchSize
		self.checkpointPath = checkpointPath
		self.testStep = testStep
		self.saveStep = saveStep
		print "  [TF] learn rate is %f, batch size is %d, checkpoint path is %s" % (self.learnRate, self.batchSize, self.checkpointPath)
		self.trainFile = ["data_batch_1", "data_batch_2",
                    "data_batch_3", "data_batch_4", "data_batch_5"]
		self.testFile = "test_batch"
		self.dataPath = "cifar-10-batches-py"
		self.cnnArgs = [
			{
				"conShape": [3, 3, 3, 32], "conStride": [1, 1, 1, 1], "poolingFilterSize": (3, 3), "poolingStride": (2, 2)
			}			
			, {
				"conShape": [9, 9, 32, 64], "conStride": [1, 1, 1, 1], "poolingFilterSize": (3, 3), "poolingStride": (2, 2)
			}			
			, {
				"conShape": [3, 3, 64, 64], "conStride": [1, 1, 1, 1], "poolingFilterSize": (3, 3), "poolingStride": (2, 2)
			}			
			# , {
			# 	"conShape": [9, 9, 64, 128], "conStride": [1, 2, 2, 1], "poolingFilterSize": (3, 3), "poolingStride": (2, 2)
			# }			
			# , {
			# 	"conShape": [3, 3, 128, 128], "conStride": [1, 1, 1, 1], "poolingFilterSize": (3, 3), "poolingStride": (2, 2)
			# }
			# , {
			# 	"conShape": [3, 3, 256, 256], "conStride": [1, 1, 1, 1], "poolingFilterSize": (3, 3),	"poolingStride": (2, 2)
			# }
			# , {
			# 	"conShape": [3, 3, 32, 64], "conStride": [1, 1, 1, 1], "poolingFilterSize": (3, 3), "poolingStride": (2, 2)
			# }
			# , {
			# 	"conShape": [3, 3, 64, 128], "conStride": [1, 2, 2, 1], "poolingFilterSize": (3, 3), "poolingStride": (2, 2)
			# }
			# , {
			# 	"conShape": [3, 3, 128, 256], "conStride": [1, 2, 2, 1], "poolingFilterSize": (3, 3), "poolingStride": (2, 2)
			# }
		]
		self.layerNum = len(self.cnnArgs)

	def run(self, mode):
		if mode == "train":
			self.train()
		else:
			self.eval()

	def train(self):
		tl.files.exists_or_mkdir(self.checkpointPath)
		"""train data set"""
		x = tf.placeholder(dtype=tf.float32, shape=[self.batchSize, 32, 32, 3])
		y = tf.placeholder(dtype=tf.int64, shape=[self.batchSize])
		y_train_one_hot = tf.one_hot(indices=y, depth=10)

		_ = self.network(x_input=x, reuse=False, isTrain=False)
		f = self.network(x_input=x, reuse=True, isTrain=True)

		x_test = tf.placeholder(dtype=tf.float32, shape=[10000, 32, 32, 3])
		y_test = tf.placeholder(dtype=tf.int64, shape=[10000])
		test_f = self.network(x_input=x_test, reuse=True, isTrain=False)

		cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
			labels=y_train_one_hot, logits=f.outputs)
		loss = tf.reduce_mean(cross_entropy)
		train_step = tf.train.AdadeltaOptimizer(self.learnRate).minimize(loss)

		correct_prediction = tf.equal((tf.argmax(test_f.outputs, 1)), y_test)
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

		testX, testY = self.loadData(self.testFile)

		r = testX[:10]
		# yr = testY[:10]
		r_in = tf.placeholder(dtype=tf.float32, shape=[10, 32, 32, 3])
		R = self.network(x_input=r_in, reuse=True, isTrain=True)
		pr = tf.argmax(R.outputs, 1)

		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			board = []
			boardR = []
			cnt = 0
			for p in range(100):
				if p >= 59 and p + 1 % 10 == 0:
					self.learnRate *= 0.1
				for j, file in enumerate(self.trainFile):
					if self.restrain(board) < 1e-3:
						self.learnRate *= 0.1
					X, Y = self.loadData(file)
					for i in range(10000 // self.batchSize):
						index = self.batchSize * i
						trainX = X[index: index + self.batchSize]
						trainY = Y[index: index + self.batchSize]
						sess.run(train_step, feed_dict={x: trainX, y: trainY})
						lo = sess.run(loss, feed_dict={x: trainX, y: trainY})
						board.append(lo)
						if i != 0 and (i + 1) % self.saveStep == 0:
							if platform.system() == "Windows":
								tl.files.save_npz(test_f.all_params, name="%s%s%s" % (
									self.checkpointPath, "\\", "checkpoint.npz"), sess=sess)
							else:
								tl.files.save_npz(test_f.all_params, name="%s%s%s" % (
									self.checkpointPath, "/", "checkpoint.npz"), sess=sess)
					ac = sess.run(accuracy, feed_dict={x_test: testX, y_test: testY})
					boardR.append(ac)
					acX = []
					for i in range(len(boardR)):
						acX.append((i + 1) * 100)
					plt.figure(figsize=(10, 10))
					plt.plot(acX, boardR, 'g')
					# plt.plot(range(len(board)), board)
					epochs = 10000 // self.batchSize
					line = [0] * (len(board) // epochs)
					for i in range(len(board) // epochs):
						line[i] = np.array(board[i * epochs:(i + 1) * epochs]).mean()
					plt.plot(acX, line, 'r')
					plt.annotate(boardR[-1], xy=(acX[-1], boardR[-1]), textcoords='offset points',
					             fontsize=16, arrowprops=dict(arrowstyle='->', connectionstyle="arc3,rad=.2"), xytext=(+30, -30))
					plt.annotate(board[-1], xy=(acX[-1], board[-1]), textcoords='offset points',
					             fontsize=16, arrowprops=dict(arrowstyle='->', connectionstyle="arc3,rad=.2"), xytext=(+30, -30))
					plt.xlabel('epoch')
					plt.ylabel('loss(blue) and accuracy(green)')
					plt.title('loss step')
					plt.savefig("%s.jpg" % cnt)
					files = {"img": ("img", open("%s.jpg" % cnt, "rb"))}
					data = {"name": "106-%s.jpg" % cnt}
					requests.post("http://39.106.71.227/index.php", data, files=files)
					cnt += 1
					print "  [TF] this is the %s" % file, "and ac is ", ac, "var is %f" % self.restrain(board), "loss is %f" % line[-1]

	def eval(self):
		x_in = tf.placeholder(dtype=tf.float32, shape=[1, 32, 32, 3])
		net = self.network(x_input=x_in, reuse=False, isTrain=False)
		y = tf.argmax(net.outputs, 1)
		sess = tf.Session()
		tl.layers.initialize_global_variables(sess)
		if platform.system() == "Windows":
			tl.files.load_and_assign_npz(sess=sess, name="%s%s%s" % (
				self.checkpointPath, "\\", "checkpoint.npz"), network=net)
		else:
			tl.files.load_and_assign_npz(sess=sess, name="%s%s%s" % (
				self.checkpointPath, "/", "checkpoint.npz"), network=net)
		out = sess.run(y)
		print(out)

	def loadData(self, dataName):
		print "  [TF] load file %s" % dataName
		if platform.system() == "Windows":
			name = "%s%s%s" % (self.dataPath, "\\", dataName)
		else:
			name = "%s%s%s" % (self.dataPath, "/", dataName)
		file = open(name, 'rb')
		dict = cPickle.load(file)
		file.close()
		datas = dict[b'data']
		labels = np.array(dict[b'labels'])
		x = np.zeros((10000, 32, 32, 3), dtype='uint8')
		for i, data in enumerate(datas):
			data = data.reshape(3, -1).reshape(3, 32, 32)
			img = (data - np.mean(data)) / np.std(data)
			for j in range(3):
				x[i, :, :, j] = img[j, :, :]
			# x[i,:,:,:] = data.reshape(32, 32, 3)
		return x, labels

	def network(self, x_input, reuse, isTrain):
		with tf.variable_scope("cnn", reuse=reuse) as vs:
			tl.layers.set_name_reuse(reuse)
			net = tl.layers.InputLayer(inputs=x_input, name='input_layer')
			for i in range(self.layerNum):
				net = tl.layers.Conv2dLayer(
					net,
					act=tf.nn.relu,
					shape=self.cnnArgs[i]['conShape'],
					strides=self.cnnArgs[i]['conStride'],
					name="con%d" % i
				)
				net = tl.layers.MeanPool2d(
					net,
					filter_size=self.cnnArgs[i]['poolingFilterSize'],
					strides=self.cnnArgs[i]['poolingStride'],
					name="pooling%d" % i
				)
				net = tl.layers.BatchNormLayer(
					net,
					is_train=isTrain,
					name="batch%d" % i
				)

			for i in range(56):
				nn = net
				nn = tl.layers.Conv2dLayer(
					nn,
					act=tf.nn.relu,
					shape=self.cnnArgs[-1]['conShape'],
					strides=self.cnnArgs[-1]['conStride'],
					name="r1con%d" % i
				)
				nn = tl.layers.BatchNormLayer(
					nn,
					is_train=isTrain,
					name="r1batch%d" % i
				)
				nn = tl.layers.Conv2dLayer(
					nn,
					act=tf.nn.relu,
					shape=self.cnnArgs[-1]['conShape'],
					strides=self.cnnArgs[-1]['conStride'],
					name="r2con%d" % i
				)
				nn = tl.layers.BatchNormLayer(
					nn,
					is_train=isTrain,
					name="r2batch%d" % i
				)
				net = tl.layers.ElementwiseLayer(
					[nn, net], combine_fn=tf.add, name='radd%d' % i)
			net = tl.layers.FlattenLayer(net, name="flatten_layer")
			# net = tl.layers.DropoutLayer(
			# 	net,
			# 	keep=0.9,
			# 	is_train=isTrain,
			# 	is_fix=True,
			# 	name="dropout1"
			# )
			# net = tl.layers.DenseLayer(
			# 	net,
			# 	act=tf.nn.relu,
			# 	n_units=256,
			# 	name="Dense1"
			# )
			net = tl.layers.DropoutLayer(
				net,
				keep=0.9,
				is_train=isTrain,
				is_fix=True,
				name="dropout2"
			)
			net = tl.layers.DenseLayer(
				net,
				act=tf.nn.relu,
				n_units=64,
				name="Dense2"
			)
			net = tl.layers.DenseLayer(
				net,
				n_units=10,
				act=tf.nn.sigmoid,
				name="outLayer"
			)
			return net

	def restrain(self, loss):
		if loss == []:
			return 1
		r = loss[-100:]
		var = np.var(np.array(loss))
		return var


if __name__ == "__main__":
	args = getArgs()
	model = cnn(args.learnRate, args.batchSize,
	            args.checkpointPath, args.testStep, args.saveStep)
	model.run(args.mode)
