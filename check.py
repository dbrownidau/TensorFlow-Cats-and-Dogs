#!/usr/bin/python3
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import input_data
import model
import os

def get_test_files(file_dir):
    tests = []
    for file in os.listdir(file_dir):
        tests.append(file_dir + file)
    print("There are %d tests" % len(tests))
    image_list = list(tests)
    return image_list

def get_one_image(image_list):
    n = len(image_list)
    ind = np.random.randint(0, n)
    img_dir = image_list[ind]

    print("Randomly selected: ", img_dir)
    image = Image.open(img_dir)
    plt.imshow(image)
    image = image.resize([208, 208])
    image = np.array(image)
    return image


def evaluate_one_image():
    '''Test one image against the saved models and parameters
     '''

    test_dir = './data/test/'
    image_array = get_one_image(get_test_files(test_dir))

    with tf.Graph().as_default():
        BATCH_SIZE = 1
        N_CLASSES = 2

        image = tf.cast(image_array, tf.float32)
        image = tf.image.per_image_standardization(image)
        image = tf.reshape(image, [1, 208, 208, 3])
        logit = model.inference(image, BATCH_SIZE, N_CLASSES)

        logit = tf.nn.softmax(logit)

        x = tf.placeholder(tf.float32, shape=[208, 208, 3])

        logs_train_dir = './logs/train/'

        saver = tf.train.Saver()

        with tf.Session() as sess:

            print("Reading checkpoints...")
            ckpt = tf.train.get_checkpoint_state(logs_train_dir)
            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)
                print("Loading success, global_step is", global_step)
            else:
                print("No checkpoint file found")

            prediction = sess.run(logit, feed_dict={x: image_array})
            max_index = np.argmax(prediction)
            if max_index == 0:
                print("This is a cat with possibility %.6f" % prediction[:, 0])
            else:
                print("This is a dog with possibility %.6f" % prediction[:, 1])

#Entrypoint
evaluate_one_image()
