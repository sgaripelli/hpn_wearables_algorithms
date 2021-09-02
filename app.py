import json
import uuid
from datetime import datetime

import pandas as pd
import numpy as np
import pytz
from flask import Flask, request
from tensorflow import keras

from preprocess import Preprocess
from kafka_interface import produce_label_to_kafka_stream, produce_log_record_to_kafka_stream
import exceptions as exc
from inference import run_inference

# Define acceptable data flags, must be lowercase
DATA_FLAGS = ['raw_sample_data', 'sleep', 'no_data', 'nodata']

ALGORITHM_NAME = 'Behavior Classification'
ALGORITHM_VERSION = '0.1'

EXPECTED_FRAME_LENGTH = 100
SENSOR_RESOLUTION = 8

LOG_TO_KAFKA = True

# Instantiate Flask object to handle REST requests
app = Flask(__name__)


def setup():
    # Hard coded for local testing should be moved to a config file for deployment
    fm_model_filepath = "bin/canine_fm_model.h5"
    sc_model_filepath = "bin/canine_ss_model.h5"

    # Instantiate preprocessor
    try:
        preprocess = Preprocess()
    except Exception as e:
        raise exc.PreprocessingError
    return fm_model_filepath, sc_model_filepath, preprocess


def load_models(fm_path, sc_path):
    try:
        fm_model = load_model(fm_path)
        sc_model = load_model(sc_path)
    except FileNotFoundError as e:
        raise e
    return fm_model, sc_model


def load_model(model_path: str):
    """ Load model using metrics described """
    return keras.models.load_model(model_path, custom_objects={
        "__f1_m": [],
        "__precision_m": [],
        "__recall_m": [],
    })


def validate_frame(data):
    timestamp_present = check_timestamp(data)
    sensorid_present = check_sensorid(data)
    data_flag_present = check_data_flag(data)
    data_present = check_data(data)
    sample_data_present = True
    if data_present:
        sample_data_present = check_sample_data(data)

    error_message = []
    if not timestamp_present:
        error_message.append('timestamp')
    if not sensorid_present:
        error_message.append('SensorID')
    if data_flag_present < 0:
        error_message.append('data_flag')
    if (not sample_data_present or not data_present) and (data_flag_present > 0):
        error_message.append('data')

    if len(error_message) > 0:
        raise exc.MissingFieldsError(error_message)


def check_data(data):
    if 'data' not in data:
        return False
    elif data.data.isnull().values:
        return False
    else:
        return True


def check_fields_present(data):
    required_fields = ['data', 'data_flag', 'SensorID', 'timestamp']
    missing_fields = set(required_fields).difference(set(data.columns))
    return missing_fields


def check_data_flag(data):
    try:
        flag = data.data_flag.iloc[0]
        if not type(flag) is str:
            return -1
        elif flag.lower() not in DATA_FLAGS:
            return -1
        elif flag.lower() == 'sleep' or flag.lower() == 'no_data' or flag.lower() == 'nodata':
            return 0
        else:
            return 1
    except AttributeError:
        return -1


def check_sample_data(data):
    required_fields = ['sampletimestamp', 'ax', 'ay', 'az', 'gx', 'gy', 'gz', 'proximity']
    sample_data = pd.DataFrame(data.data.iloc[0])
    if not sample_data.isnull().values.any():
        if len(set(required_fields).difference(set(sample_data.columns))) > 0:
            return False
        try:
            validate_data_fields(sample_data)
        except exc.MissingFieldsError:
            return False
        return True
    else:
        return exc.BadDataError('Sample Data Missing Required Fields')


def check_timestamp(data):
    try:
        timestamp_str2millis(data.timestamp.iloc[0])
    except (ValueError, AttributeError):
        return False
    else:
        return True


def check_sensorid(data):
    try:
        if data.SensorID.iloc[0] == '' or data.SensorID.iloc[0] == 'MISSING' or pd.isnull(data.SensorID.iloc[0]):
            return False
        return True
    except AttributeError:
        return False


def validate_data_fields(data_df):
    try:
        [timestamp_str2millis(sampletimestamp) for sampletimestamp in data_df.sampletimestamp]
    except ValueError:
        raise exc.BadDataError('Cannot interpret sample timestamps')
    if not all([type(val) is float for signal in zip(data_df.ax, data_df.ay, data_df.az) for val in signal]):
        try:
            [pd.to_numeric(val) for signal in zip(data_df.ax, data_df.ay, data_df.az) for val in signal]
        except Exception as e:
            raise exc.BadDataError('Cannot interpret signal data')


def check_sensorid_timestamp(data, i_frame):
    try:
        timestamp_epoch = timestamp_str2millis(data.timestamp[i_frame])
    except (AttributeError, ValueError, TypeError):
        # If timestamp is not included set to default
        timestamp_epoch = 0

    try:
        sensor_id = data.SensorID[i_frame]
        if pd.isnull(sensor_id):
            raise AttributeError
        else:
            sensor_id = str(sensor_id)
    except AttributeError:
        sensor_id = 'MISSING'

    return timestamp_epoch, sensor_id


def load_data(json_to_load):
    if type(json_to_load) is pd.DataFrame:
        try:
            data = json_to_load
        except ValueError as e:
            raise e
    else:
        try:
            data = pd.DataFrame.from_dict(json_to_load)
        except ValueError as e:
            raise e
    return data


def process_raw_sample_data(preprocess, fm_model, sc_model, data, i_frame, sensor_id, timestamp_epoch):
    sample_output = []
    # Check number of samples in frame. For CMAS and earlier we expect 100, HPN1 expects 111.
    sample = data.data[i_frame]

    # Instantiate class prediction label, accelerometery matrix and inference data dataframe
    class_prediction = None

    # Get accelerometer data
    try:
        x_axis = [float(row.get('ax')) / SENSOR_RESOLUTION for row in sample]
        y_axis = [float(row.get('ay')) / SENSOR_RESOLUTION for row in sample]
        z_axis = [float(row.get('az')) / SENSOR_RESOLUTION for row in sample]
    except TypeError as e:
        raise exc.BadDataError('Missing Axis')
    except ValueError as e:
        raise exc.BadDataError('Missing Data')

    # All axes should have the same length

    if True in [None in [x, y, z] for [x, y, z] in zip(x_axis, y_axis, z_axis)] or \
            True in [any([x == "", y == "", z == ""]) for [x, y, z] in zip(x_axis, y_axis, z_axis)]:
        raise exc.BadDataError('Axes are not equal lengths')
    elif len(x_axis) < EXPECTED_FRAME_LENGTH:
        # Record No data for frame
        sample_output = {'sensorid': sensor_id, 'sampledatetimestamp': timestamp_epoch, 'activity': 'PARTIAL_DATA'}
    else:
        if len(x_axis) > EXPECTED_FRAME_LENGTH:
            # If more samples than expected are present, trim and classify as normal
            x_axis = x_axis[0:EXPECTED_FRAME_LENGTH]
            y_axis = y_axis[0:EXPECTED_FRAME_LENGTH]
            z_axis = z_axis[0:EXPECTED_FRAME_LENGTH]

        try:
            # Build accelerometer data matrix and flatten to 300x1 window
            acc = pd.DataFrame({'ax': x_axis, 'ay': y_axis, 'az': z_axis})
            acc = acc.apply(pd.to_numeric)
            filtered_data = preprocess.filter(acc)
            flat_data = np.asarray(filtered_data).flatten('F').reshape(1, -1)

            # Reshape and Normalize data
            infer_data_norm = preprocess.normalise(flat_data)
            stacked_data = preprocess.stack_data(infer_data_norm)
            inference_data = preprocess.reshape_data(stacked_data)
        except ValueError:
            raise exc.PreprocessingError
        # Run inference using specified model
        if acc is not None and inference_data is not None:
            try:
                class_prediction = run_inference(flat_data, inference_data, sc_model, fm_model)
            except Exception:
                raise exc.InferenceError
        if class_prediction is not None:
            # If class prediction has been successfully set, append prediction to sample output
            sample_output = {'sensorid': sensor_id, 'sampledatetimestamp': timestamp_epoch,
                             'activity': class_prediction}

    return sample_output


def log_error(payload_uuid, data, msg, created_dt, status='Failure'):
    """ Return error, record log """
    log_record = format_error_log(payload_uuid, data, msg, created_dt, 'Failure')
    if LOG_TO_KAFKA:
        produce_log_record_to_kafka_stream(json.dumps(log_record))
    # If the whole processing pipeline fails return the output, otherwise just log the error as a partial failure
    if status == 'Failure':
        return json.dumps({'success': False, 'test_output': msg}), 200, {
            'ContentType': 'application/json'}


def format_and_log_frame_error(data, i_frame, label, error_message, message):
    timestamp_epoch, sensor_id = check_sensorid_timestamp(data, i_frame)
    sample_output = {'sensorid': sensor_id, 'sampledatetimestamp': timestamp_epoch, 'activity': label}
    error_message.append(i_frame, message)
    return sample_output


def format_error_log(log_uuid, data, msg, created_dt, status):
    """ Format log record. If no data is passed in use default values """
    if data is None:
        log_record = {'algorithm_uuid': str(log_uuid),
                      'algorithm_name': ALGORITHM_NAME,
                      'algorithm_version': ALGORITHM_VERSION,
                      'device_number': 'MISSING',
                      'sample_start_time': timestamp_str2millis('1970-01-01 00:00:00'),
                      'sample_end_time': timestamp_str2millis('1970-01-01 00:00:00'),
                      'error_log': msg,
                      'created_date': created_dt,
                      'algorithm_status': status,
                      'modified': True}
    else:
        if 'SensorID' not in data:
            log_sensor_id = 'MISSING'
        else:
            log_sensor_id = str(data.SensorID[0])
        if 'timestamp' not in data:
            log_start_timestamp = '1970-01-01 00:00:00'
            log_end_timestamp = '1970-01-01 00:00:00'
        else:
            log_start_timestamp = data.timestamp[0]
            log_end_timestamp = data.timestamp.iloc[-1]
            try:
                timestamp_str2millis(log_start_timestamp)
            except (ValueError, TypeError):
                log_start_timestamp = '1970-01-01 00:00:00'
            try:
                timestamp_str2millis(log_end_timestamp)
            except (ValueError, TypeError):
                log_end_timestamp = '1970-01-01 00:00:00'
        log_record = {'algorithm_uuid': str(log_uuid),
                      'algorithm_name': ALGORITHM_NAME,
                      'algorithm_version': ALGORITHM_VERSION,
                      'device_number': log_sensor_id,
                      'sample_start_time': timestamp_str2millis(log_start_timestamp),
                      'sample_end_time': timestamp_str2millis(log_end_timestamp),
                      'error_log': msg,
                      'created_date': created_dt,
                      'algorithm_status': status,
                      'modified': True}
    return log_record


def format_payload_for_error(data: pd.DataFrame) -> pd.DataFrame:
    """ Format a new payload to account for missing data """
    present_cols = data.columns
    output_df_schema = data
    if 'data_flag' not in present_cols:
        output_df_schema['data_flag'] = 'MISSING'
    if 'SensorID' not in present_cols:
        output_df_schema['SensorID'] = 'MISSING'
    if 'timestamp' not in present_cols:
        output_df_schema['timestamp'] = '1970-01-01 00:00:00'
    else:
        try:
            timestamp_str2millis(output_df_schema['timestamp'][0])
        except ValueError:
            output_df_schema['timestamp'] = '1970-01-01 00:00:00'
    if 'data' not in present_cols:
        output_df_schema['data'] = None

    return pd.DataFrame(output_df_schema)


def timestamp_str2millis(timestamp) -> int:
    if isinstance(timestamp, int) or isinstance(timestamp, float) or isinstance(timestamp, np.int64):
        return int(timestamp)
    else:
        """" Convert a string based timestamp to Unix timestamp """
        if timestamp.isnumeric():
            return int(timestamp)
        if len(timestamp) == 19:
            return int(
                (datetime.strptime(str(timestamp), "%Y-%m-%d %H:%M:%S") - datetime(1970, 1, 1)).total_seconds()) * 1000
        elif len(timestamp) == 23:
            return int(
                (datetime.strptime(str(timestamp), "%Y-%m-%d %H:%M:%S.%f") - datetime(1970, 1,
                                                                                      1)).total_seconds() * 1000)
        else:
            raise ValueError


@app.route("/classify", methods=["POST"])
def infer():
    # Set created timestamp
    created_dt = timestamp_str2millis(datetime.now(pytz.timezone('UTC')).strftime("%Y-%m-%d %H:%M:%S"))

    # Instantiate unique id for payload processing
    payload_uuid = uuid.uuid4()

    fm_model_filepath, sc_model_filepath, preprocess = setup()
    # Load Model for inference
    fm_model, sc_model = load_models(fm_model_filepath, sc_model_filepath)
    try:
        data = load_data(request.get_json())
    except ValueError as e:
        return log_error(payload_uuid, format_payload_for_error(e.data),
                         'Unable to parse JSON: {}'.format(str(e)), created_dt)

    # Get number of frames to be inferred
    n_frames = len(data)

    # Instantiate output string
    test_output = []

    # Instantiate error message to aggregate errors
    e_msg = ErrorMsg()

    # insert processing record into algorithm log table
    log_record = format_error_log(payload_uuid, data, 'n/a', created_dt, 'InProgress')
    if LOG_TO_KAFKA:
        produce_log_record_to_kafka_stream(json.dumps(log_record))

    all_frames_output = []
    # For each sample
    for i_frame in range(n_frames):
        # Instantiate empty list to hold output data
        sample_output = []
        sample_data = data
        try:
            validate_frame(pd.DataFrame(data.iloc[i_frame]).transpose())
        except exc.MissingFieldsError as e:
            error_log = format_and_log_frame_error(sample_data, i_frame,
                                                   'GAP', e_msg,
                                                   'FAILED_PROCESSING.missing_fields: {}'.format(str(e)))
            sample_output.append(error_log)
        except exc.BadDataError as e:
            sample_output.append(
                format_and_log_frame_error(sample_data, i_frame, 'GAP', e_msg,
                                           'FAILED_PROCESSING.bad_data: {}'.format(str(e))))
        else:
            # Get timestamp of sample
            timestamp_epoch = timestamp_str2millis(data.timestamp[i_frame])
            sensor_id = str(data.SensorID[i_frame])

            # Conditional to check sample type. If Raw_Sample_Data process inference, if NO_DATA or SLEEP return
            # unchanged
            if data.data_flag[i_frame].lower() == 'Raw_Sample_Data'.lower():
                try:
                    sample_output.append(
                        process_raw_sample_data(preprocess, fm_model, sc_model, data, i_frame, sensor_id,
                                                timestamp_epoch))
                except exc.PreprocessingError:
                    # if exception is raised during preprocessing, log error and skip frame
                    sample_output.append(format_and_log_frame_error(sample_data, i_frame,
                                                                    'GAP', e_msg,
                                                                    'FAILED_PROCESSING.preprocessing'))
                except exc.BadDataError:
                    sample_output.append(
                        format_and_log_frame_error(sample_data, i_frame, 'GAP', e_msg,
                                                   'FAILED_PROCESSING.preprocessing'))

                except exc.InferenceError:
                    # if exception is raised during inference, log error and skip frame
                    sample_output.append(format_and_log_frame_error(sample_data, i_frame, 'GAP',
                                                                    e_msg, 'FAILED_PROCESSING.inference'))
            elif data.data_flag[i_frame].lower() == "NO_DATA".lower() or data.data_flag[i_frame].lower() == 'nodata':
                sample_output.append(
                    {'sensorid': sensor_id, 'sampledatetimestamp': timestamp_epoch, 'activity': 'NODATA'})

            elif data.data_flag[i_frame].lower() == "Sleep".lower():
                sample_output.append(
                    {'sensorid': sensor_id, 'sampledatetimestamp': timestamp_epoch, 'activity': 'Sleep'})
            else:
                # If data flag is not recognised return error, record log
                sample_output.append(format_and_log_frame_error(sample_data, i_frame, 'GAP',
                                                                e_msg, 'FAILED_PROCESSING.data_flag'))

        test_output.append(sample_output)
        all_frames_output.append(sample_output)
    if LOG_TO_KAFKA:
        produce_label_to_kafka_stream(json.dumps(all_frames_output))

    if len(e_msg) == n_frames:
        log_record = format_error_log(payload_uuid, data, str(e_msg), created_dt, 'Failure')
    elif len(e_msg) > 0:
        log_record = format_error_log(payload_uuid, data, str(e_msg), created_dt, 'Partial_Failure')
    else:
        log_record = format_error_log(payload_uuid, data, str(e_msg), created_dt, 'Success')
    if LOG_TO_KAFKA:
        produce_log_record_to_kafka_stream(json.dumps(log_record))

    return json.dumps({'success': True, 'test_output': test_output, 'errors': str(e_msg)}), 200, {'ContentType': 'application/json'}


class ErrorMsg:
    def __init__(self):
        self.msg = {}

    def __str__(self):
        return json.dumps(self.msg)

    def __len__(self):
        return len(self.msg)

    def append(self, frame_idx, error_message):
        self.msg[frame_idx] = error_message


if __name__ == "__main__":
    app.run(host="localhost", port=8080, debug=True)
