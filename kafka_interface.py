from confluent_kafka import SerializingProducer
from confluent_kafka.schema_registry import SchemaRegistryClient
from confluent_kafka.schema_registry.avro import AvroSerializer
from confluent_kafka.serialization import StringSerializer
import json
import datetime
import pytz
import uuid

config_file = 'config/kafka.conf'
classifier_schema_registry_conf = {
    'url': 'https://psrc-4rw99.us-central1.gcp.confluent.cloud',
    'basic.auth.user.info': 'XZSDXKTVFY675FLY:IElkQsBgfP6FKLHWQoCgaRiO3pTTQQtS302ElwR+dWl7UD5JG/Z9rMbZKXccrWZy'
}


def produce_log_record_to_kafka_stream(log_record):
    schema_file = 'config/algorithm_log_schema.avro'
    log_topic = 'B_ALGORITHM_LOG'

    with open(config_file) as file:
        conf = json.loads(file.read().replace('\n', ''))

    schema_registry_client = SchemaRegistryClient(classifier_schema_registry_conf)

    # read log schema
    with open(schema_file) as file:
        log_schema = file.read().replace('\n', '')

    value_avro_serializer = AvroSerializer(str(log_schema),
                                           schema_registry_client,
                                           LogObj.log_value_to_dict)

    producer_conf = {
        'bootstrap.servers': conf['bootstrap.servers'],
        'sasl.mechanisms': conf['sasl.mechanisms'],
        'security.protocol': conf['security.protocol'],
        'sasl.username': conf['sasl.username'],
        'sasl.password': conf['sasl.password'],
        'key.serializer': StringSerializer(),
        'value.serializer': value_avro_serializer,
        'batch.size': 100000,
        'linger.ms': 10}
    producer = SerializingProducer(producer_conf)

    data = json.loads(log_record)
    value_object = LogObj(algorithm_uuid=data['algorithm_uuid'],
                          algorithm_name=data['algorithm_name'],
                          algorithm_version=data['algorithm_version'],
                          device_number=data['device_number'],
                          sample_start_time=data['sample_start_time'],
                          sample_end_time=data['sample_end_time'],
                          algorithm_status=data['algorithm_status'],
                          created_date=data['created_date'],
                          error_log=data['error_log'],
                          modified=data['modified'])

    producer.produce(topic=log_topic, key=data['algorithm_uuid'], value=value_object, on_delivery=acked)
    producer.poll(0)
    producer.flush()


def produce_label_to_kafka_stream(output):
    schema_file = 'config/activity_label_schema.avro'
    classifier_topic = 'activity_label'

    # read kafka config
    with open(config_file) as file:
        conf = json.loads(file.read().replace('\n', ''))

    schema_registry_client = SchemaRegistryClient(classifier_schema_registry_conf)

    # read activity label schema
    with open(schema_file) as file:
        activity_label_schema = file.read().replace('\n', '')

    value_avro_serializer = AvroSerializer(str(activity_label_schema),
                                           schema_registry_client,
                                           ActivityObj.activity_value_to_dict)

    producer_conf = {
        'bootstrap.servers': conf['bootstrap.servers'],
        'sasl.mechanisms': conf['sasl.mechanisms'],
        'security.protocol': conf['security.protocol'],
        'sasl.username': conf['sasl.username'],
        'sasl.password': conf['sasl.password'],
        'key.serializer': StringSerializer(),
        'value.serializer': value_avro_serializer,
        'batch.size': 100000,
        'linger.ms': 10}
    producer = SerializingProducer(producer_conf)

    data = json.loads(output)

    for value in data:
        value = value[0]
        value_object = ActivityObj(sensorid=value['sensorid'], activity=value['activity'],
                                   sampledatetimestamp=value['sampledatetimestamp'])
        producer.produce(topic=classifier_topic, key=value['sensorid'], value=value_object, on_delivery=acked)
        producer.poll(0)

    producer.flush()

class ActivityObj(object):

    # utility class to return a dictionary of algorithm output
    # used as the value by producer.produce
    def __init__(self, sensorid, activity, sampledatetimestamp):
        self.sensorid = sensorid

        self.activity = activity
        self.sampledatetimestamp = sampledatetimestamp

    def activity_value_to_dict(value_obj, ctx):
        return dict(sensorid=value_obj.sensorid,
                    activity=value_obj.activity,
                    sampledatetimestamp=value_obj.sampledatetimestamp)


class LogObj(object):

    def __init__(self, algorithm_uuid, algorithm_name, algorithm_version, device_number, sample_start_time,
                 sample_end_time, algorithm_status, created_date, error_log, modified):
        self.algorithm_uuid = uuid.UUID(algorithm_uuid).bytes
        self.algorithm_name = algorithm_name
        self.algorithm_version = algorithm_version
        self.device_number = device_number
        self.sample_start_time = sample_start_time
        self.sample_end_time = sample_end_time
        self.algorithm_status = algorithm_status
        self.error_log = error_log
        self.created_by = 'RJTAnalytics'
        self.created_date = created_date
        self.modified_by = 'RJTAnalytics'
        self.modified_date = datetime.datetime.now(pytz.timezone('UTC'))

    def log_value_to_dict(self, ctx):
        return (dict(algorithm_uuid=self.algorithm_uuid,
                     algorithm_name=self.algorithm_name,
                     algorithm_version=self.algorithm_version,
                     device_number=self.device_number,
                     sample_start_time=self.sample_start_time,
                     sample_end_time=self.sample_end_time,
                     algorithm_status=self.algorithm_status,
                     error_log=self.error_log,
                     created_by=self.created_by,
                     created_date=self.created_date,
                     modified_by=self.modified_by,
                     modified_date=self.modified_date
                     ))


def acked(err, msg):
    """Delivery report handler called on
    successful or failed delivery of message
    """

    if err is not None:
        raise Exception("Failed to deliver message: {}, msg {}".format(err, msg))
