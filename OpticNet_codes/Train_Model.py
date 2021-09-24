from OpticNet.load_data import load_data
import tensorflow as tf
import time
from tensorflow.keras.optimizers import Adam


def train(model, base_dir, epoch, cb):
    train_batches, test_batches, val_batches, classes = load_data(base_dir)
    start_time = time.time()
    # Training the model
    batch_size=16
    history = model.fit(train_batches, 
                    shuffle=True,  
                    validation_data=val_batches, 
                    epochs=epoch, 
                    verbose=1, 
                    callbacks=[cb])

    end_time = time.time()
    print("--- Time taken to train : %s hours ---" % ((end_time - start_time) // 3600))

    return history



