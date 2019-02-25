from dataset import DataSet
from model import TitanicModel
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier

epochs = 2500

data = DataSet(
    '/home/manoolia/code/python/kaggle/titanic-challange/input/train.csv',
    '/home/manoolia/code/python/kaggle/titanic-challange/input/test.csv'
)

data.load_data()

model = TitanicModel()

layer0 = tf.nn.relu(model.fully_connected_layer(model.x_train, size=240))

drop0 = tf.nn.dropout(tf.cast(layer0, dtype=tf.float32), model.hold_prop)

layer1 = tf.nn.relu(model.fully_connected_layer(drop0, size=240))

drop1 = tf.nn.dropout(tf.cast(layer1, dtype=tf.float32), model.hold_prop)

layer2 = tf.nn.relu(model.fully_connected_layer(drop1, size=160))

drop2 = tf.nn.dropout(tf.cast(layer2, dtype=tf.float32), model.hold_prop)

layer3 = tf.nn.relu(model.fully_connected_layer(drop2, size=80))

output_layer = tf.nn.softmax(model.fully_connected_layer(layer3, size=2))


cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=model.y_train, logits=output_layer))

optimizer = tf.train.AdamOptimizer(learning_rate=0.001)

train = optimizer.minimize(cross_entropy)

pred = tf.cast(tf.greater_equal(output_layer, 0.8), tf.float64, name='pred')  # 1 if >= 0.5

acc = tf.reduce_mean(tf.cast(tf.equal(pred, model.y_train), tf.float64), name='acc')

init = tf.global_variables_initializer()


with tf.Session() as sess:

    sess.run(init)

    for i in range(epochs):

        batch = data.next_batch(32)

        sess.run(train, feed_dict={model.x_train: batch[0], model.y_train: batch[1],  model.hold_prop: 0.2})

        if i % 50 == 0:

            train_acc = sess.run(acc, feed_dict={model.x_train: data.test, model.y_train: data.test_labels, model.hold_prop: 1})

            print("Test Acc: {}".format(train_acc))





    prediction = sess.run(pred, feed_dict={model.x_train: data.production, model.hold_prop:1})


    df = data.test_df
    df["Survived"] = data.rev_one_hot_encode(prediction)
    df = df[['PassengerId', 'Survived']]
    df = df.set_index("PassengerId")
    df.to_csv("/home/manoolia/code/python/kaggle/titanic-challange/input/test_result_2.csv", sep=",")

