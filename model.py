import time
import tensorflow as tf
 
data = tf.constant([[1,1],[0,0],[2,0],[2,1],[3,0],[0,4],[5,6],[0,10]])
label = tf.constant([1,0,0,1,0,0,1,0])


model = tf.keras.Sequential(
    [
        tf.keras.Input(shape=(2,)),
        tf.keras.layers.Dense(20, activation='relu'),
        tf.keras.layers.Dense(1, activation='softmax')
    ]
)

print(model.summary())

model.compile(optimizer='adam', 
              loss=tf.keras.losses.BinaryCrossentropy(), 
              metrics=['accuracy'])

model.fit(data, label, batch_size=2, epochs=5)
 
save_time = int(time.time())
Path = f'./saved_models/{save_time}'
model.save(Path, save_format='tf')

