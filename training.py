#!/usr/bin/python3

import os
import numpy as np
import tensorflow as tf
import input_data #Import functions from input_data.py
import model #Import functions from model.py

N_CLASSES = 2
IMG_W = 208  # resize the image
IMG_H = 208
BATCH_SIZE = 16 # Number of images to process per step
CAPACITY = 2000
MAX_STEP = 10000  # Maximum steps
learning_rate = 0.0001  # passed to the Adam optimizer (https://www.tensorflow.org/api_docs/python/tf/train/AdamOptimizer)

def main():

    print("**")
    print("* Batch Size: %d" % BATCH_SIZE)
    print("* Capacity: %d" % CAPACITY)
    print("* Maximum Steps: %d" % MAX_STEP)
    print("* Learning Rate: %d" % learning_rate)
    print("**")
    print()

    train_dir = './data/train/'
    logs_train_dir = './logs/train/'

    print("Loading dataset..")
    (train, train_label) = input_data.get_files(train_dir)

    print("Loading batch...")
    (train_batch, train_label_batch) = input_data.get_batch(
        train,
        train_label,
        IMG_W,
        IMG_H,
        BATCH_SIZE,
        CAPACITY,
        )

    print("Constructing inference...")
    train_logits = model.inference(train_batch, BATCH_SIZE, N_CLASSES)
    print("Constructing losses...")
    train_loss = model.losses(train_logits, train_label_batch)
    print("Constructing training...")
    train_op = model.trainning(train_loss, learning_rate)
    print("Constructing evaluation...")
    train__acc = model.evaluation(train_logits, train_label_batch)

    summary_op = tf.summary.merge_all()
    sess = tf.Session()
    train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)
    saver = tf.train.Saver()

    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
        for step in np.arange(MAX_STEP):
            if coord.should_stop():
                break
            (_, tra_loss, tra_acc) = sess.run([train_op, train_loss, train__acc])

            if step % 50 == 0:
                print ("Step %d, train loss = %.2f, train accuracy = %.2f%%" % (step, tra_loss, tra_acc * 100.0))
                summary_str = sess.run(summary_op)
                train_writer.add_summary(summary_str, step)

            if step % 2000 == 0 or step + 1 == MAX_STEP:
                print("Saving model at step %d" % step)
                checkpoint_path = os.path.join(logs_train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)
    except tf.errors.OutOfRangeError:
        print("Done training -- epoch limit reached")
    finally:
        print("Shutting down...")
        coord.request_stop()

    coord.join(threads)
    sess.close()
    exit()


#Entrypoint
main()
