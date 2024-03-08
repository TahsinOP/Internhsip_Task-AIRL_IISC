#!/usr/bin/env python3

import tensorflow as tf

class PointNetModel:
    
    def __init__(self):
        
        self.model = tf.keras.models.load_model('/src')

    def predict(self, data):
        # Assuming data is preprocessed and in the correct format
        return self.model.predict(data)