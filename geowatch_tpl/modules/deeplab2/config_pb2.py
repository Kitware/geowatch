# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: deeplab2/config.proto

from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database

# @@protoc_insertion_point(imports)


from deeplab2 import dataset_pb2 as deeplab2_dot_dataset__pb2
from deeplab2 import evaluator_pb2 as deeplab2_dot_evaluator__pb2
from deeplab2 import model_pb2 as deeplab2_dot_model__pb2
from deeplab2 import trainer_pb2 as deeplab2_dot_trainer__pb2

# from deeplab2.dataset_pb2 import *
# from deeplab2.evaluator_pb2 import *
# from deeplab2.model_pb2 import *
from deeplab2.model_pb2 import DecoderOptions, ModelOptions

# to pass unused lint
var = ModelOptions() != DecoderOptions()

_sym_db = _symbol_database.Default()
DESCRIPTOR = _descriptor.FileDescriptor(
    name='deeplab2/config.proto',
    package='deeplab2',
    syntax='proto2',
    serialized_options=b'P\001',
    serialized_pb=b'\n\x15\x64\x65\x65plab2/config.proto\x12\x08\x64\x65\x65plab2\x1a\x16\x64\x65\x65plab2/dataset.proto\x1a\x18\x64\x65\x65plab2/evaluator.proto\x1a\x14\x64\x65\x65plab2/model.proto\x1a\x16\x64\x65\x65plab2/trainer.proto\"\xb6\x02\n\x11\x45xperimentOptions\x12\x17\n\x0f\x65xperiment_name\x18\x01 \x01(\t\x12-\n\rmodel_options\x18\x02 \x01(\x0b\x32\x16.deeplab2.ModelOptions\x12\x31\n\x0ftrainer_options\x18\x03 \x01(\x0b\x32\x18.deeplab2.TrainerOptions\x12\x37\n\x15train_dataset_options\x18\x04 \x01(\x0b\x32\x18.deeplab2.DatasetOptions\x12\x35\n\x11\x65valuator_options\x18\x05 \x01(\x0b\x32\x1a.deeplab2.EvaluatorOptions\x12\x36\n\x14\x65val_dataset_options\x18\x06 \x01(\x0b\x32\x18.deeplab2.DatasetOptionsB\x02P\x01P\x00P\x01P\x02P\x03'
    ,
    dependencies=[deeplab2_dot_dataset__pb2.DESCRIPTOR, deeplab2_dot_evaluator__pb2.DESCRIPTOR,
                  deeplab2_dot_model__pb2.DESCRIPTOR, deeplab2_dot_trainer__pb2.DESCRIPTOR, ],
    public_dependencies=[deeplab2_dot_dataset__pb2.DESCRIPTOR, deeplab2_dot_evaluator__pb2.DESCRIPTOR,
                         deeplab2_dot_model__pb2.DESCRIPTOR, deeplab2_dot_trainer__pb2.DESCRIPTOR, ])

_EXPERIMENTOPTIONS = _descriptor.Descriptor(
    name='ExperimentOptions',
    full_name='deeplab2.ExperimentOptions',
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    fields=[
        _descriptor.FieldDescriptor(
            name='experiment_name', full_name='deeplab2.ExperimentOptions.experiment_name', index=0,
            number=1, type=9, cpp_type=9, label=1,
            has_default_value=False, default_value=b"".decode('utf-8'),
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            serialized_options=None, file=DESCRIPTOR),
        _descriptor.FieldDescriptor(
            name='model_options', full_name='deeplab2.ExperimentOptions.model_options', index=1,
            number=2, type=11, cpp_type=10, label=1,
            has_default_value=False, default_value=None,
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            serialized_options=None, file=DESCRIPTOR),
        _descriptor.FieldDescriptor(
            name='trainer_options', full_name='deeplab2.ExperimentOptions.trainer_options', index=2,
            number=3, type=11, cpp_type=10, label=1,
            has_default_value=False, default_value=None,
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            serialized_options=None, file=DESCRIPTOR),
        _descriptor.FieldDescriptor(
            name='train_dataset_options', full_name='deeplab2.ExperimentOptions.train_dataset_options', index=3,
            number=4, type=11, cpp_type=10, label=1,
            has_default_value=False, default_value=None,
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            serialized_options=None, file=DESCRIPTOR),
        _descriptor.FieldDescriptor(
            name='evaluator_options', full_name='deeplab2.ExperimentOptions.evaluator_options', index=4,
            number=5, type=11, cpp_type=10, label=1,
            has_default_value=False, default_value=None,
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            serialized_options=None, file=DESCRIPTOR),
        _descriptor.FieldDescriptor(
            name='eval_dataset_options', full_name='deeplab2.ExperimentOptions.eval_dataset_options', index=5,
            number=6, type=11, cpp_type=10, label=1,
            has_default_value=False, default_value=None,
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            serialized_options=None, file=DESCRIPTOR),
    ],
    extensions=[
    ],
    nested_types=[],
    enum_types=[
    ],
    serialized_options=None,
    is_extendable=False,
    syntax='proto2',
    extension_ranges=[],
    oneofs=[
    ],
    serialized_start=132,
    serialized_end=442,
)

_EXPERIMENTOPTIONS.fields_by_name['model_options'].message_type = deeplab2_dot_model__pb2._MODELOPTIONS
_EXPERIMENTOPTIONS.fields_by_name['trainer_options'].message_type = deeplab2_dot_trainer__pb2._TRAINEROPTIONS
_EXPERIMENTOPTIONS.fields_by_name['train_dataset_options'].message_type = deeplab2_dot_dataset__pb2._DATASETOPTIONS
_EXPERIMENTOPTIONS.fields_by_name['evaluator_options'].message_type = deeplab2_dot_evaluator__pb2._EVALUATOROPTIONS
_EXPERIMENTOPTIONS.fields_by_name['eval_dataset_options'].message_type = deeplab2_dot_dataset__pb2._DATASETOPTIONS
DESCRIPTOR.message_types_by_name['ExperimentOptions'] = _EXPERIMENTOPTIONS
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

ExperimentOptions = _reflection.GeneratedProtocolMessageType('ExperimentOptions', (_message.Message,), {
    'DESCRIPTOR': _EXPERIMENTOPTIONS,
    '__module__': 'deeplab2.config_pb2'
    # @@protoc_insertion_point(class_scope:deeplab2.ExperimentOptions)
})
_sym_db.RegisterMessage(ExperimentOptions)

DESCRIPTOR._options = None
# @@protoc_insertion_point(module_scope)