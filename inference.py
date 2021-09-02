import numpy as np
import pandas as pd

MAGNITUDE_THRESHOLD = 0.05

# Define class labels
FM_CLASSES = ['Not Recognised as Forward Motion', 'Walking', 'Trotting', 'Cantering', 'Galloping']
SC_CLASSES = ['Not Scratch/Shake', 'Scratching', 'Shaking']


def run_inference(acc, inference_data, sc_model, fm_model):
    """ Run the inference using prescribed models """
    # First sorting by magnitude
    signal_magnitude = calculate_mean_magnitude(acc)
    if signal_magnitude < MAGNITUDE_THRESHOLD:
        class_prediction = 'Resting'
    elif signal_magnitude > 1.5:
        class_prediction = 'Not Recognised FM or S/S'
    else:
        class_probabilities = sc_model.predict_step(inference_data)
        # If model has low confidence label pass to fm classifier, otherwise label according to highest #
        # probability
        if np.max(class_probabilities) < 0.75 or np.argmax(class_probabilities) == 0:
            class_probabilities = fm_model.predict_step(inference_data)
            if np.max(class_probabilities) < 0.75:
                if signal_magnitude < 0.1:
                    class_prediction = 'Walking'
                elif signal_magnitude < 0.2:
                    class_prediction = 'Trotting'
                elif signal_magnitude < 0.35:
                    class_prediction = 'Cantering'
                elif signal_magnitude < 0.5:
                    class_prediction = 'Galloping'
                elif signal_magnitude > 0.5:
                    class_prediction = 'Not Recognised FM or S/S'
            else:
                class_prediction = FM_CLASSES[np.argmax(class_probabilities)]
        else:
            class_prediction = SC_CLASSES[np.argmax(class_probabilities)]
    return class_prediction


def calculate_mean_magnitude(accel: pd.DataFrame) -> float:
    """ Calculate the signal magnitude for use assigning models to frames """
    x = accel[:, 300:400]
    y = accel[:, 400:500]
    z = accel[:, 500:600]
    return float(np.mean(np.sqrt(np.square(x) + np.square(y) + np.square(z))))
    # return float(np.mean(np.sum(np.abs(accel.loc[300:600].values), 1)))
