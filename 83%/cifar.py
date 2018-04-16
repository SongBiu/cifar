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



os.environ["CUDA_VISIBLE_DEVICES"] = "3"

def getArgs():
	parser = argparse.ArgumentParser()
	parser.add_argument("-m", "--mode", help="choose train or eval", type=str, default="train", choices=["train", "eval"])
	parser.add_argument("-l", "--learnRate", help="input the learn rate", type=float, default=1.)
	parser.add_argument("-b", "--batchSize", help="input the batch size", type=int, default=100)
	parser.add_argument("-p", "--checkpointPath", help="input the path of checkpoint", type=str, default="ckp")
	parser.add_argument("-t", "--testStep", help="the step of test", type=int, default=10)
	parser.add_argument("-s", "--saveStep", help="the step of save checkpoint", type=int, default=500)
	args = parser.parse_args()
	return args


class cnn:
	def __init__(self, learnRate, batchSize, checkpointPath, testStep, saveStep):
		self.learnRate = learnRate
		self.batchSize = batchSize
		self.checkpointPath = checkpointPath
		self.testStep = testStep
		self.saveStep = saveStep
		self.trainFiles = ["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5"]
		self.testFile = "test_batch"
		self.dataPath = "cifar-10-batches-py"
		
	def run(self, mode):
		if mode == "train":
			self.train()
		else:
			self.eval()

	def buildargs(self):
		self.x = tf.placeholder(dtype=tf.float32, shape=[self.batchSize, 32, 32, 3])
		self.y = tf.placeholder(dtype=tf.int64, shape=[self.batchSize])
		self.y_train_one_hot = tf.one_hot(indices=self.y, depth=10)
		self._ = self.network(x_input=self.x, reuse=False, isTrain=False)
		self.f = self.network(x_input=self.x, reuse=True, isTrain=True)
		self.x_test = tf.placeholder(dtype=tf.float32, shape=[10000, 32, 32, 3])
		self.y_test = tf.placeholder(dtype=tf.int64, shape=[10000])
		self.test_f = self.network(x_input=self.x_test, reuse=True, isTrain=False)
		self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.y_train_one_hot, logits=self.f.outputs)
		self.loss = tf.reduce_mean(self.cross_entropy)
		self.correct_prediction = tf.equal((tf.argmax(self.test_f.outputs, 1)), self.y_test)
		self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, "float"))
		self.lossLine = []
		self.accuracyTestLine = []
		self.accuracyTrainLine = []
		self.lossMean = []
		self.epochSize = 10000 // self.batchSize
		self.epoch = 1
		self.step = 0

	# def savecheckpoint(self):
		

	def train(self):
		tl.files.exists_or_mkdir(self.checkpointPath)
		self.buildargs()
		train_step = tf.train.AdadeltaOptimizer(self.learnRate).minimize(self.loss)
		testX, testY = self.loadData(self.testFile)
		config = tf.ConfigProto()
		config.gpu_options.per_process_gpu_memory_fraction = 0.8
		with tf.Session(config=config) as sess:
			sess.run(tf.global_variables_initializer())
			for i in range(500):
				if i >= 59 and  i + 1 % 100 == 0 and self.learnRate >= 1e-6:
					self.learnRate *= 0.1
				for file in self.trainFiles:
					X, Y = self.loadData(file)
					for j in range(10000//self.batchSize):
						index = self.batchSize * j
						trainX = X[index: index + self.batchSize]
						trainY = Y[index: index + self.batchSize]
						sess.run(train_step, feed_dict={self.x: trainX, self.y: trainY})
						lossPoint = sess.run(self.loss, feed_dict={self.x: trainX, self.y:trainY})
						self.lossLine.append(lossPoint)
						if self.step != 0 and (self.step + 1) % self.saveStep == 0:
							if platform.system() == "Windows":
								tl.files.save_npz(self.test_f.all_params, name="%s%s%s" % (self.checkpointPath, "\\", "checkpoint.npz"), sess=sess)
							else:
								tl.files.save_npz(self.test_f.all_params, name="%s%s%s" % (self.checkpointPath, "/", "checkpoint.npz"), sess=sess)
						self.step += 1
					accuracyTestPoint = sess.run(self.accuracy, feed_dict={self.x_test: testX, self.y_test: testY})
					accuracyTrainPoint = sess.run(self.accuracy, feed_dict={self.x_test: X, self.y_test: Y})					
					self.accuracyTestLine.append(accuracyTestPoint)
					self.accuracyTrainLine.append(accuracyTrainPoint)
					print "[TF] this is the %d epochs, %s" % (i, file), "and accuracy on trainis ", accuracyTrainPoint, "accuracy on test is", accuracyTestPoint, "var is %f" % np.var(np.array(self.lossLine)), "loss is %f" % self.lossLine[-1]
					self.lossMean.append(np.array(self.lossLine[-self.epochSize:]).mean())
					self.drawAndpost()
					
					
	def drawAndpost(self):
		xDisc = np.array(range(self.epoch+1)[1:]) * self.epochSize
		xCon = np.array(range(self.epoch * self.epochSize))
		plt.figure(figsize=(10, 10))
		plt.plot(xCon, self.lossLine, 'b')
		plt.plot(xDisc, self.accuracyTrainLine, 'y')
		plt.plot(xDisc, self.accuracyTestLine, 'g')
		plt.plot(xDisc, self.lossMean, 'r')
		plt.plot(xCon, [1.0] * self.epoch * self.epochSize, 'r--')
		plt.annotate(self.accuracyTrainLine[-1], xy=(xDisc[-1], self.accuracyTrainLine[-1]), textcoords='offset points', fontsize=16, arrowprops=dict(arrowstyle='->', connectionstyle="arc3,rad=.2"), xytext=(+30, +30)) 
		plt.annotate(self.accuracyTestLine[-1], xy=(xDisc[-1], self.accuracyTestLine[-1]), textcoords='offset points', fontsize=16, arrowprops=dict(arrowstyle='->', connectionstyle="arc3,rad=.2"), xytext=(+30, -30)) 
		plt.annotate(self.lossMean[-1], xy=(xDisc[-1], self.lossMean[-1]), textcoords='offset points', fontsize=16, arrowprops=dict(arrowstyle='->', connectionstyle="arc3,rad=.2"), xytext=(+30, -30)) 
		plt.xlabel('epoch')
		plt.ylabel('loss(blue) and accuracy(green)')
		plt.title('loss step')
		plt.savefig("img/%s.jpg" % self.epoch)
		filebin = open("img/%s.jpg" % self.epoch, "rb")
		files = {"img": ("img", filebin)}
		data = {"name": "108-%s.jpg" % self.epoch}
		requests.post("http://39.106.71.227/index.php", data, files=files)
		filebin.close()
		self.epoch += 1

	def eval(self):
		x_in = tf.placeholder(dtype=tf.float32, shape=[1, 32, 32, 3])
		net = self.network(x_input=x_in, reuse=False, isTrain=False)
		y = tf.argmax(net.outputs, 1)
		sess = tf.Session()
		tl.layers.initialize_global_variables(sess)
		if platform.system() == "Windows":
			tl.files.load_and_assign_npz(sess=sess, name="%s%s%s" % (self.checkpointPath, "\\", "checkpoint.npz"), network=net)
		else:
			tl.files.load_and_assign_npz(sess=sess, name="%s%s%s" % (self.checkpointPath, "/", "checkpoint.npz"), network=net)
		out = sess.run(y)
		print(out)

	def loadData(self, dataName):
		print "[TF] load file %s" % dataName
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
			for j in range(3):		
				x[i, :, :, j] = data[j, :, :]
			# x[i,:,:,:] = data
		return x, labels

	def network(self, x_input, reuse, isTrain):
		with tf.variable_scope("cnn", reuse=reuse) as vs:
			tl.layers.set_name_reuse(reuse)
			net = tl.layers.InputLayer(inputs=x_input, name='input_layer')
			"""block 1"""
			net = tl.layers.Conv2dLayer(net, shape=[3, 3, 3, 32], strides=[1, 1, 1, 1], act=tf.nn.relu, padding="SAME", name="b1c1")
			net = tl.layers.Conv2dLayer(net, shape=[3, 3, 32, 64], strides=[1, 1, 1, 1], act=tf.nn.relu, padding="SAME", name="b1c2")
			net = tl.layers.MaxPool2d(net, filter_size=(2, 2), strides=(2, 2), padding="SAME", name="b1p")
			net = tl.layers.BatchNormLayer(net, is_train=isTrain, act=tf.nn.relu, name="b1b")
			
			"""block 2"""
			net = tl.layers.Conv2dLayer(net, shape=[3, 3, 64, 128], strides=[1, 1, 1, 1], act=tf.nn.relu, padding="SAME", name="b2c1")
			# net = tl.layers.Conv2dLayer(net, shape=[3, 3, 48, 64], strides=[1, 1, 1, 1], act=tf.nn.relu, name="b2c2")
			net = tl.layers.Conv2dLayer(net, shape=[3, 3, 128, 128], strides=[1, 1, 1, 1], act=tf.nn.relu, padding="SAME", name="b2c3")
			net = tl.layers.MaxPool2d(net, filter_size=(2, 2), strides=(2, 2), padding="SAME", name="b2p")
			net = tl.layers.BatchNormLayer(net, is_train=isTrain, act=tf.nn.relu, name="b2b")
			# net = tl.layers.DropoutLayer(net, keep=0.9, is_train=isTrain, is_fix=True, name="dropoutp1")
			
			
			"""block 3"""
			# net = tl.layers.Conv2dLayer(net, shape=[3, 3, 32, 64], strides=[1, 1, 1, 1], act=tf.nn.relu, padding="SAME", name="b3c1")
			# net = tl.layers.Conv2dLayer(net, shape=[3, 3, 48, 64], strides=[1, 1, 1, 1], act=tf.nn.relu, name="b3c2")
			# net = tl.layers.Conv2dLayer(net, shape=[3, 3, 64, 128], strides=[1, 1, 1, 1], act=tf.nn.relu, padding="SAME", name="b3c3")
			# net = tl.layers.BatchNormLayer(net, is_train=isTrain, act=tf.nn.relu, name="b3b")
			# net = tl.layers.MaxPool2d(net, filter_size=(2, 2), strides=(2, 2), padding="SAME", name="b3p")

			"""residual block"""
			for i in range(3):
				nn = net
				nn = tl.layers.Conv2dLayer(nn, act=tf.nn.relu, shape=[3, 3, 128, 128], strides=[1, 1, 1, 1], padding="SAME", name="r1con%d" % i)
				nn = tl.layers.BatchNormLayer(nn, is_train=isTrain, act=tf.nn.relu, name="r1batch%d" % i)
				nn = tl.layers.MeanPool2d(nn, filter_size=(2, 2), strides=(1, 1), padding="SAME", name="r1%dp" % i)
				# nn = tl.layers.Conv2dLayer(nn, act=tf.nn.relu, shape=[3, 3, 128, 128], strides=[1, 1, 1, 1], padding="SAME", name="r2con%d" % i)
				# nn = tl.layers.BatchNormLayer(nn, is_train=isTrain, act=tf.nn.relu, name="r2batch%d" % i)
				# nn = tl.layers.MeanPool2d(nn, filter_size=(2, 2), strides=(1, 1), padding="SAME", name="r2%dp" % i)
				net = tl.layers.ElementwiseLayer([nn, net], combine_fn=tf.add, name='radd%d' % i)
			"""Dense Layers"""
			net = tl.layers.FlattenLayer(net, name="flatten_layer")
			net = tl.layers.DenseLayer(net, act=tf.nn.relu, n_units=256, name="Dense1")
			net = tl.layers.DropoutLayer(net, keep=0.9, is_train=isTrain, is_fix=True, name="dropout1")
			net = tl.layers.DenseLayer(net, act=tf.nn.relu, n_units=64, name="Dense2")
			net = tl.layers.DropoutLayer(net, keep=0.9, is_train=isTrain, is_fix=True, name="dropout2")
			# net = tl.layers.DenseLayer(net, act=tf.nn.relu ,n_units=32, name="Dense3")
			# net = tl.layers.DropoutLayer(net, keep=0.9, is_train=isTrain, is_fix=True, name="dropout3")
			net = tl.layers.DenseLayer(net, n_units=10, act=tf.nn.sigmoid, name="outLayer")
			return net



if __name__ == "__main__":
	args = getArgs()
	model = cnn(args.learnRate, args.batchSize, args.checkpointPath, args.testStep, args.saveStep)
	model.run(args.mode)
