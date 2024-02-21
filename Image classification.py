import tensorflow as tf
import matplotlib.pyplot as plt

class mycallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy')>0.9):
            print("\nReached 90 accuracy so cancelling training")
            self.model.stop_training=True

callbacks=mycallback()
mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images,test_labels) = mnist.load_data()

plt.imshow(training_images[0])
plt.show()
print(training_images[0])
print(training_labels[0])

training_images = training_images/255.0
test_images = test_images/255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer=tf.keras.optimizers.Adam(), loss="sparse_categorical_crossentropy", metrics=['accuracy'])
model.fit(training_images, training_labels, epochs=5, callbacks=(callbacks))
model.evaluate(test_images,test_labels)

classification=model.predict(test_images)
print(classification[0])
print(test_labels[0])
