# YOLO3 (Detection, Training, and Evaluation)

> This project is based on: https://github.com/experiencor/keras-yolo3

## Training guide:

#### 1. Data preparation 

Download the Raccoon dataset from from https://github.com/experiencor/raccoon_dataset.

#### 2. Edit the configuration file
You have to modify the parameters in the file `config.json`: ```labels```, ```train_image_folder```, ```train_annot_folder```, ```cache_name```.
To modify other parameters are optional.
The configuration file is a json file, which looks like this:
```json
{
    "model" : {
        "min_input_size":       352,
        "max_input_size":       448,
        "anchors":              [117,142, 151,233, 187,341, 245,377, 248,223, 288,302, 324,379, 374,274, 383,387],
        "labels":               ["raccoon"]
    },

    "train": {
        "train_image_folder":   "/home/minhnc-lab/WORKSPACES/AI/data/raccoon_dataset/images/",
        "train_annot_folder":   "/home/minhnc-lab/WORKSPACES/AI/data/raccoon_dataset/annotations/",
        "cache_name":           "raccoon_train.pkl",

        "train_times":          10,
        "pretrained_weights":   "",
        "batch_size":           4,
        "learning_rate":        1e-4,
        "nb_epochs":             50,
        "warmup_epochs":        3,
        "ignore_thresh":        0.5,
        "gpus":                 "0",

        "grid_scales":          [1,1,1],
        "obj_scale":            5,
        "noobj_scale":          1,
        "xywh_scale":           1,
        "class_scale":          1,

        "tensorboard_dir":      "logs",
        "saved_weights_name":   "yolo3.weights",
        "debug":                true
    },

    "valid": {
        "valid_image_folder":   "",
        "valid_annot_folder":   "",
        "cache_name":           "",

        "valid_times":          1
    }
}
```

Download pretrained weights for backend at: http://www.mediafire.com/file/l1b96fk7j18yi7v/backend.h5

**This weights must be put in the root folder of the repository. They are the pretrained weights for the backend only and will be loaded during model creation. The code does not work without this weights.**

#### 3. Generate anchors for your dataset (optional)

`python gen_anchors.py -c config.json`

Copy the generated anchors printed on the terminal to the ```anchors``` setting in ```config.json```.

#### 4. Start the training process

`python train.py -c config.json`

By the end of this process, the code will write the weights of the best model to file best_weights.h5 (or whatever name specified in the setting "saved_weights_name" in the config.json file). The training process stops when the loss on the validation set is not improved in 3 consecutive epoches.

#### 5. Perform detection using trained weights on image, set of images, video, or webcam
`python predict.py -c config.json -i /path/to/image/or/video`

It carries out detection on the image and write the image with detected bounding boxes to the same folder.

## Evaluation

`python evaluate.py -c config.json`

Compute the mAP performance of the model defined in `saved_weights_name` on the validation dataset defined in `valid_image_folder` and `valid_annot_folder`.
