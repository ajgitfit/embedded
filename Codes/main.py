#https://www.tensorflow.org/addons/tutorials/optimizers_cyclicallearningrate
# installation files
pip install tensorflow
pip install scipy
pip install tensorflow-addons
pip install -q -U tensorflow_addons

def get_training_model():
    model = tf.keras.Sequential(
        [
            layers.Input((100, 100, 3)),
            layers.experimental.preprocessing.Rescaling(scale=1./255),
            layers.Conv2D(16, (5, 5), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(32, (5, 5), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.SpatialDropout2D(0.2),
            layers.GlobalAvgPool2D(),
            layers.Dense(128, activation="relu"),
            layers.Dense(1, activation="softmax"),
        ]
    )
    return model

def train_model(model, optimizer):
    model.compile(loss="categorical_crossentropy", optimizer=optimizer,
                       metrics=["accuracy"])
    history = model.fit(x_train,
        y_train,
        batch_size=BATCH_SIZE,
        validation_data=(x_test, y_test),
        epochs=EPOCHS)
    return history

initial_model = get_training_model()
initial_model.save("my_model")
standard_model = tf.keras.models.load_model("initial_model")
no_clr_history = train_model(standard_model, optimizer="sgd")
