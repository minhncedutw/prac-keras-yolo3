'''
    File name: keras-yolo3
    Author: minhnc-lab
    Date created(MM/DD/YYYY): 2019-05-03
    Last modified(MM/DD/YYYY HH:MM): 2019-05-03 10:28
    Python Version: 3.6
    Other modules: [None]

    Copyright = Copyright (C) 2017 of CONG-MINH NGUYEN
    Credits = [None] # people who reported bug fixes, made suggestions, etc. but did not actually write the code
    License = None
    Version = 0.9.0.1
    Maintainer = [None]
    Email = minhnc.edu.tw@gmail.com
    Status = Prototype # "Prototype", "Development", or "Production"
    Project Style: https://dev.to/codemouse92/dead-simple-python-project-structure-and-imports-38c6
    Code Style: http://web.archive.org/web/20111010053227/http://jaynes.colorado.edu/PythonGuidelines.html#module_formatting
'''

#==============================================================================
# Imported Modules
#==============================================================================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import sys
import time

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # The GPU id to use, usually either "0" or "1"

import json
import numpy as np

import tensorflow as tf
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from callbacks import CustomModelCheckpoint, CustomTensorBoard
from keras.models import load_model

from voc import parse_voc_annotation
from generator import BatchGenerator
from utils.utils import normalize, makedirs, evaluate
from utils.multi_gpu_model import multi_gpu_model
from yolo_re import create_yolov3_model, dummy_loss

#==============================================================================
# Constant Definitions
#==============================================================================

#==============================================================================
# Function Definitions
#==============================================================================
def create_training_instances(train_annot_folder, train_image_folder, train_cache,
                              valid_annot_folder, valid_image_folder, valid_cache,
                              labels):
    # parse annotations of the training set
    train_ints, train_labels = parse_voc_annotation(ann_dir=train_annot_folder, img_dir=train_image_folder, cache_name=train_cache, labels=labels)

    # parse annotations of the validation set, if any, otherwise split the training set
    if os.path.exists(valid_annot_folder):
        valid_ints, valid_labels = parse_voc_annotation(valid_annot_folder, valid_image_folder, valid_cache, labels)
    else:
        print("valid_annot_folder not exists. Spliting the trainining set.")

        train_valid_split = int(0.8 * len(train_ints))
        np.random.seed(seed=0)
        np.random.shuffle(train_ints)
        np.random.seed(seed=None)

        valid_ints = train_ints[train_valid_split:]
        train_ints = train_ints[:train_valid_split]

    # compare the seen labels with the given labels in config.json
    if len(labels) > 0:
        overlap_labels = set(labels).intersection(set(train_labels.keys()))

        print('Seen labels: \t' + str(train_labels))
        print('Given labels: \t' + str(labels))

        # return None, None, None if some given label is not in the dataset
        if len(overlap_labels) < len(labels):
            print('Some labels have no annotations! Please revise the list of labels in the config.json.')
            return None, None, None
    else:
        print('No labels are provided. Train on all seen labels.')
        print(train_labels)
        labels = train_labels.keys()

    max_box_per_image = max([len(inst['object']) for inst in (train_ints + valid_ints)])

    return train_ints, valid_ints, sorted(labels), max_box_per_image


def create_model(
        nb_class,
        anchors,
        max_box_per_image,
        max_grid,
        batch_size,
        warmup_batches,
        ignore_thresh,
        multi_gpu,
        saved_weights_name,
        lr,
        grid_scales,
        obj_scale,
        noobj_scale,
        xywh_scale,
        class_scale
):
    if multi_gpu > 1:
        with tf.device('/cpu:0'):
            template_model, infer_model = create_yolov3_model(
                nb_class            =nb_class,
                anchors             =anchors,
                max_box_per_image   =max_box_per_image,
                max_grid            =max_grid,
                batch_size          =batch_size//multi_gpu,
                warmup_batches      =warmup_batches,
                ignore_thresh       =ignore_thresh,
                grid_scales         =grid_scales,
                obj_scale           =obj_scale,
                noobj_scale         =noobj_scale,
                xywh_scale          =xywh_scale,
                class_scale         =class_scale
            )
    else:
        template_model, infer_model = create_yolov3_model(
            nb_class            =nb_class,
            anchors             =anchors,
            max_box_per_image   =max_box_per_image,
            max_grid            =max_grid,
            batch_size          =batch_size,
            warmup_batches      =warmup_batches,
            ignore_thresh       =ignore_thresh,
            grid_scales         =grid_scales,
            obj_scale           =obj_scale,
            noobj_scale         =noobj_scale,
            xywh_scale          =xywh_scale,
            class_scale         =class_scale
        )

    # load the pretrained weight if exists, otherwise load the backend weight only
    if os.path.exists(saved_weights_name):
        print("\nLoading pretrained weights.\n")
        template_model.load_weights(saved_weights_name)
    else:
        template_model.load_weights("backend.h5", by_name=True)

    train_model = template_model

    optimizer = Adam(lr=lr, clipnorm=0.001)
    train_model.compile(loss=dummy_loss, optimizer=optimizer)

    return train_model, infer_model


def create_callbacks(saved_weights_name, tensorboard_logs, model_to_save):
    makedirs(path=tensorboard_logs)

    early_stop = EarlyStopping(
        monitor     ='loss',
        min_delta   =0.01,
        patience    =5,
        mode        ='min',
        verbose     =1
    )

    checkpoint = CustomModelCheckpoint(
        model_to_save   =model_to_save,
        filepath        =saved_weights_name,
        monitor         ='loss',
        verbose         =1,
        save_best_only  =True,
        mode            ='min',
        period          =1
    )
    reduce_on_plateau = ReduceLROnPlateau(
        monitor     ='loss',
        factor      =0.1,
        patience    =2,
        verbose     =1,
        mode        ='min',
        epsilon     =0.01,
        cooldown    =0,
        min_lr      =0
    )
    tensorboard = CustomTensorBoard(
        log_dir         =tensorboard_logs,
        write_graph     =True,
        write_images    =True
    )
    return [early_stop, checkpoint, reduce_on_plateau, tensorboard]

#==============================================================================
# Main function
#==============================================================================
def _main_(args):
    print('Hello World! This is {:s}'.format(args.desc))

    config_path = args.conf
    with open(config_path) as config_buffer:
        config = json.loads(config_buffer.read())

    ###############################
    #   Prepare the data
    ###############################
    ## Parse the annotations
    train_ints, valid_ints, labels, max_box_per_image = create_training_instances(
        config['train']['train_annot_folder'],
        config['train']['train_image_folder'],
        config['train']['cache_name'],
        config['valid']['valid_annot_folder'],
        config['valid']['valid_image_folder'],
        config['valid']['cache_name'],
        config['model']['labels']
    )
    print('\nTraining on: \t' + str(labels) + '\n')

    ## Create the generators
    train_generator = BatchGenerator(
        instances           =train_ints,
        anchors             =config['model']['anchors'],
        labels              =labels,
        downsample          =32,
        max_box_per_image   =max_box_per_image,
        batch_size          =config['train']['batch_size'],
        min_net_size        =config['model']['min_input_size'],
        max_net_size        =config['model']['max_input_size'],
        shuffle             =True,
        jitter              =0.3,
        norm                =normalize
    )

    valid_generator = BatchGenerator(
        instances           =valid_ints,
        anchors             =config['model']['anchors'],
        labels              =labels,
        downsample          =32,
        max_box_per_image   =max_box_per_image,
        batch_size          =config['train']['batch_size'],
        min_net_size        =config['model']['min_input_size'],
        max_net_size        =config['model']['max_input_size'],
        shuffle             =True,
        jitter              =0.0,
        norm                =normalize # /255
    )

    ###############################
    #   Create the model
    ###############################
    if os.path.exists(config['train']['saved_weights_name']):
        config['train']['warmup_epochs'] = 0
    warmup_batches = config['train']['warmup_epochs'] * (config['train']['train_times']*len(train_generator))

    os.environ['CUDA_VISIBLE_DEVICES'] = config['train']['gpus']
    multi_gpu = len(config['train']['gpus'].split(','))

    train_model, infer_model = create_model(
        nb_class            =len(labels),
        anchors             =config['model']['anchors'],
        max_box_per_image   =max_box_per_image,
        max_grid            =[config['model']['max_input_size'], config['model']['max_input_size']],
        batch_size          =config['train']['batch_size'],
        warmup_batches      =warmup_batches,
        ignore_thresh       =config['train']['ignore_thresh'],
        multi_gpu           =multi_gpu,
        saved_weights_name  =config['train']['saved_weights_name'],
        lr                  =config['train']['learning_rate'],
        grid_scales         =config['train']['grid_scales'],
        obj_scale           =config['train']['obj_scale'],
        noobj_scale         =config['train']['noobj_scale'],
        xywh_scale          =config['train']['xywh_scale'],
        class_scale         =config['train']['class_scale']
    )

    ###############################
    #   Kick off the training
    ###############################
    callbacks = create_callbacks(saved_weights_name    =config['train']['saved_weights_name'],
                                  tensorboard_logs      =config['train']['tensorboard_dir'],
                                  model_to_save         =infer_model
    )

    train_model.fit_generator(
        generator=train_generator,
        steps_per_epoch=len(train_generator) * config['train']['train_times'],
        epochs=config['train']['nb_epochs'] + config['train']['warmup_epochs'],
        verbose=2 if config['train']['debug'] else 1,
        callbacks=callbacks,
        workers=4,
        max_queue_size=8
    )

    ###############################
    #   Run the evaluation
    ###############################
    # compute mAP for all the classes
    average_precisions = evaluate(infer_model, valid_generator)

    # print the score
    for label, average_precision in average_precisions.items():
        print(labels[label] + ': {:.4f}'.format(average_precision))
    print('mAP: {:.4f}'.format(sum(average_precisions.values()) / len(average_precisions)))


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Your program name!!!')
    argparser.add_argument('-d', '--desc', help='description of the program', default='redo the keras-yolo3')
    argparser.add_argument('-c', '--conf', default='config.json', help='path to configuration file')

    args = argparser.parse_args()
    _main_(args)
