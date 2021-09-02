import pandas as pd
from datetime import datetime
import numpy as np

DATA_FLAGS = ['raw_sample_data', 'sleep', 'no_data']

class MissingFieldsError(Exception):
    """ incorrectly formatted data is passed to function """
    def __init__(self, missing_fields):
        # present_cols = data.columns
        self.message = 'Invalid data, missing: '
        missing_fields_string = ' '.join(missing_fields)
        self.message += missing_fields_string

    def __str__(self):
        return self.message
    pass


class MissingDataFlagError(Exception):
    """ data_flag is not present """
    def __str__(self):
        return 'Missing Data Flags'
    pass


class MissingSensorIdError(Exception):
    """ SensorID is not present """
    def __str__(self):
        return 'Missing SensorIDs'
    pass


class MissingTimestampError(Exception):
    """ Timestamp is not present """
    def __str__(self):
        return 'Missing or incorrect timestamps'
    pass


class BadDataError(Exception):
    """ Sensor Data is mis-formatted or wrong type"""
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return 'Sensor Data is mis-formatted or wrong type: {}'.format(self.message)
    pass


class PreprocessingError(Exception):
    """Preprocessing Failed"""
    pass


class InferenceError(Exception):
    """Inference Failed"""
    pass
