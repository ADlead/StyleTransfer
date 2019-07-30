import numpy as np
import tensorflow as tf
import os

from PIL import Image
import time

''' 全局参数 '''
vgg16_npy_path = './data/vgg16.npy'
content_img_path = './data/hai2++.jpg'
style_img_path = './data/xingkong.jpg'
output_path = './ouput'

VGG_MEAN = [103.939, 116.779, 123.68]  # ??
lambda_c = 0.01
lambda_s = 100

learning_rate = 10
num_step = 1000


# 搭建vggnet的结构，把模型文件中的参数导入进来
class VGGNet:
    def __init__(self, data_dict):
        self.data_dict = data_dict

    # 获取卷积层参数
    def get_conv_filter(self, name):
        return tf.constant(self.data_dict[name][0], name='conv')

    # 获得全连接层w
    def get_fc_weight(self, name):
        return tf.constant(self.data_dict[name][0], name='fc')

    # 获取卷积层偏置
    def get_bias(self, name):
        return tf.constant(self.data_dict[name][1], name='bias')

    # 卷积实现
    def conv_layer(self, x, name):
        with tf.name_scope(name):
            conv_w = self.get_conv_filter(name)
            conv_b = self.get_bias(name)
            # 卷积操作
            h = tf.nn.conv2d(x, conv_w, [1,1,1,1], padding='SAME')
            h = tf.nn.bias_add(h, conv_b)
            h = tf.nn.relu(h)
            return h

    #  池化层实现
    def pooling_layer(self, x, name):
        return tf.nn.max_pool(x, ksize=[1,2,2,1],
                              strides=[1,2,2,1],
                              padding='SAME',
                              name=name)

    # 全连接层实现
    def fc_layer(self, x, name, activation=tf.nn.relu):
        with tf.name_scope(name):
            fc_w = self.get_fc_weight(name)
            fc_b = self.get_bias(name)
            h = tf.matmul(x, fc_w)
            h = tf.nn.bias_add(h, fc_b)
            if activation is None:
                return h
            else:
                return activation(h)

    # 展平实现
    def flatten_layer(self, x, name):
        with tf.name_scope(name):
            x_shape = x.get_shape().as_list()
            dim = 1
            for d in x_shape[1:]:
                dim *= d
            x = tf.reshape(x, [-1, dim])
            return x

    # 构建网络
    def build(self, x_rgb):
        start_time = time.time()
        print('building model ...')
        r, g, b = tf.split(x_rgb, [1,1,1], axis=3) # 第四个维度
        x_bgr = tf.concat([
            b - VGG_MEAN[0],
            g - VGG_MEAN[1],
            r - VGG_MEAN[2]],
            axis = 3
        )

        assert  x_bgr.get_shape().as_list()[1:] == [224, 224, 3]

        self.conv1_1 = self.conv_layer(x_bgr, 'conv1_1')
        self.conv1_2 = self.conv_layer(self.conv1_1, 'conv1_2')
        self.pool1 = self.pooling_layer(self.conv1_2, 'pool1')

        self.conv2_1 = self.conv_layer(self.pool1, 'conv2_1')
        self.conv2_2 = self.conv_layer(self.conv2_1, 'conv2_2')
        self.pool2 = self.pooling_layer(self.conv2_2, 'pool2')

        self.conv3_1 = self.conv_layer(self.pool2, 'conv3_1')
        self.conv3_2 = self.conv_layer(self.conv3_1, 'conv3_2')
        self.conv3_3 = self.conv_layer(self.conv3_2, 'conv3_3')
        self.pool3 = self.pooling_layer(self.conv3_3, 'pool3')

        self.conv4_1 = self.conv_layer(self.pool3, 'conv4_1')
        self.conv4_2 = self.conv_layer(self.conv4_1, 'conv4_2')
        self.conv4_3 = self.conv_layer(self.conv4_2, 'conv4_3')
        self.pool4 = self.pooling_layer(self.conv4_3, 'pool4')

        self.conv5_1 = self.conv_layer(self.pool4, 'conv5_1')
        self.conv5_2 = self.conv_layer(self.conv5_1, 'conv5_2')
        self.conv5_3 = self.conv_layer(self.conv5_2, 'conv5_3')
        self.pool5 = self.pooling_layer(self.conv5_3, 'pool5')

        # 图像风格转换，可以不使用全连接层

        print('building model finished %4ds' % (time.time() - start_time))



# 初始化结果图像
def initial_result(shape, mean, stddev):
    initial = tf.truncated_normal(shape, mean=mean, stddev=stddev)
    return tf.Variable(initial)

def read_img(img_name):
    img = Image.open(img_name)
    np_img = np.array(img) # 224 224 3
    np_image = np.asarray([np_img], dtype=np.int32) # 1 224 224 3
    return np_image

# gram矩阵的计算
def gram_matrix(x):
    b, w, h, ch = x.get_shape().as_list()
    features = tf.reshape(x, [b, h*w, ch])
    gram = tf.matmul(features, features, adjoint_a=True) \
    / tf.constant(ch*w*h, tf.float32)
    return gram

result = initial_result((1, 224, 224, 3), 127.5, 20)
content_val = read_img(content_img_path)
style_val = read_img(style_img_path)

content = tf.placeholder(tf.float32, shape=[1, 224, 224, 3], name='content')
style = tf.placeholder(tf.float32, shape=[1, 224, 224, 3], name='style')

# 为内容图像、风格图像、结果图像 分别创建vggnet
data_dict = np.load(vgg16_npy_path, encoding='latin1', allow_pickle=True).item()
vgg_for_content = VGGNet(data_dict)
vgg_for_style = VGGNet(data_dict)
vgg_for_result = VGGNet(data_dict)

# 创建每个图像的vggnet
vgg_for_content.build(content)
vgg_for_style.build(style)
vgg_for_result.build(result)

content_features = [
    vgg_for_content.conv1_2,
    #vgg_for_content.conv2_2
]

result_content_features = [
    vgg_for_result.conv1_2,
    #vgg_for_result.conv2_2
]

# 根据层数定制文件夹名称
conv_len = len(result_content_features)
output_dir = output_path + '-%d' % (conv_len)
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

style_features = [
    #vgg_for_style.conv2_2,
    vgg_for_style.conv3_3
]

result_style_features = [
    #vgg_for_result.conv2_2,
    vgg_for_result.conv3_3
]

# 风格损失的gram矩阵计算
style_gram = [gram_matrix(feature) for feature in style_features]
result_style_gram = [gram_matrix(feature) for feature in result_style_features]

# 计算内容损失
content_loss = tf.zeros(1, tf.float32)
for c, c_ in zip(content_features, result_content_features):
    content_loss += tf.reduce_mean((c-c_)**2, [1,2,3])

# 计算风格损失
style_loss = tf.zeros(1, tf.float32)
for s, s_ in zip(style_gram, result_style_gram):
    style_loss += tf.reduce_mean((s-s_)**2, [1,2])

# 内容损失和风格损失 加权
loss = content_loss * lambda_c + style_loss * lambda_s
train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# tensorflow 框架
init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)
    for step in range(num_step):
        loss_value, content_loss_value, style_loss_value, _ = sess.run(
            [loss, content_loss, style_loss, train_op],
            feed_dict={
                content: content_val,
                style: style_val
            }
        )
        print('step: %d, loss_value: %8.4f, content_loss: %8.4f, style_loss: %8.4f' % \
              (step+1, loss_value[0],
               content_loss_value[0], style_loss_value[0]))

        result_img_path = os.path.join(output_dir, 'result-%05d.jpg' % (step+1))
        result_val = result.eval(sess)[0] # 取出变量
        result_val = np.clip(result_val, 0, 255)
        img_arr = np.asarray(result_val, np.uint8)
        img = Image.fromarray(img_arr)
        img.save(result_img_path)
