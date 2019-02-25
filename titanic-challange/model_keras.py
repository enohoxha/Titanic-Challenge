from dataset import DataSet
from tensorflow.contrib import keras
from tensorflow.contrib.keras import layers, losses, models, activations, optimizers
epochs = 800

data = DataSet(
    '/home/manoolia/code/python/kaggle/titanic-challange/input/train.csv',
    '/home/manoolia/code/python/kaggle/titanic-challange/input/test.csv'
)

data.load_data()


model = models.Sequential()

model.add(layers.Dense(
    units=240,
    activation=activations.sigmoid,
    input_shape=[5,]
))
model.add(layers.Dropout(rate=0.2))
model.add(layers.BatchNormalization())

model.add(layers.Dense(
    units=160,
    activation=activations.relu,
))
model.add(layers.Dropout(rate=0.2))
model.add(layers.BatchNormalization())
model.add(layers.Dense(
    units=80,
    activation=activations.sigmoid,
))
model.add(layers.Dropout(rate=0.2))
model.add(layers.BatchNormalization())

model.add(layers.Dense(
    units=2,
    activation=activations.softmax,
))

model.compile(optimizer=optimizers.Adam(lr=0.001), loss=losses.categorical_crossentropy, metrics=['accuracy'])

model.fit(x=data.train, y=data.train_labels, batch_size=32, epochs=epochs, verbose=2)

print("Test")

_, acc = model.evaluate(data.test, data.test_labels)

print(acc)