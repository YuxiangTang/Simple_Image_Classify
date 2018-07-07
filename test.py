# =============================================================================
from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import module.CNN_Model as CNN_Model
from module.Norm_Batch import get_files
import os


# =======================================================================
# 获取一张图片
def get_one_image(train):
    # 输入参数：train,训练图片的路径
    # 返回参数：image，从训练图片中随机抽取一张图片
    n = len(train)
    ind = np.random.randint(0, n)
    img_dir = train[ind]  # 随机选择测试的图片

    img = Image.open(img_dir)
    plt.imshow(img)
    imag = img.resize([64, 64])  # 由于图片在预处理阶段以及resize，因此该命令可略
    image = np.array(imag)
    return image


# --------------------------------------------------------------------
# 测试图片
def evaluate_one_image(image_array, image_label):
    # you need to change the directories to yours.
    BATCH_SIZE = 1
    N_CLASSES = 4
    x = tf.placeholder(tf.float32, shape=[64, 64, 3])
    imaget = tf.cast(x, tf.float32)
    imaget = tf.image.per_image_standardization(imaget)
    imaget = tf.reshape(imaget, [1, 64, 64, 3])
    logit = CNN_Model.inference(imaget, BATCH_SIZE, N_CLASSES,1)
    # logit = tf.nn.softmax(logit)
    n = len(image_array)
    target = 0
    saver = tf.train.Saver()
    with tf.Session() as sess:
        logs_train_dir = os.getcwd() + '/logs'
        print("Reading checkpoints...")
        ckpt = tf.train.get_checkpoint_state(logs_train_dir)
        if ckpt and ckpt.model_checkpoint_path:
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('Loading success, global_step is %s' % global_step)
        else:
            print('No checkpoint file found')
        for count in range(n):
            img_dir = image_array[count]
            img = Image.open(img_dir)
            imag = img.resize([64, 64])
            image = np.array(imag)
            prediction = sess.run(logit, feed_dict={x: image})
            max_index = np.argmax(prediction)
            if max_index==image_label[count]:
                target+=1
    acc = target/n
    print('accuracy: %.6f' % acc)

# ------------------------------------------------------------------------

if __name__ == '__main__':
    test_dir = os.getcwd()+'/data/InputData/test'
    test, test_label, _, _ = get_files(test_dir, 0)
    #img = get_one_image(test)  # 通过改变参数train or val，进而验证训练集或测试集
    evaluate_one_image(test, test_label)
# ===========================================================================
