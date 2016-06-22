# Finding Action Tubes

Source code for Finding Action Tubes, created by Georgia Gkioxari at UC Berkeley.

[![Gitter](https://badges.gitter.im/gkioxari/ActionTubes.svg)](https://gitter.im/gkioxari/ActionTubes?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)

### Contents
1. [Requirements](#requirements)
2. [Usage](#usage)
3. [Instructions](#instructions)
4. [Downloads](#downloads)

### Requirements

You need `Caffe`. Please refer to [Caffe  installation instructions](http://caffe.berkeleyvision.org/installation.html)

### Usage

The pipeline consists of multiple steps. For simplicity, we break them down to independent procedures. 

1. In `startup.m`, add the paths and resolve the dependencies

2. Selective search boxes need to be stored in the following format  
    `/ss_dir/motion/action/video/0000f.mat`  
  Video frames need to be stored in the following format  
    `/img_dir/action/video/0000f.png`

3. Optical flow computation

    `compute_OF/compute_flow.m`  
      for a pair of images computes flow as described in the paper.  
        im1, im2: input images  
        optical flow images need to be stored in the format `/flow_dir/action/video/0000f.png`

4. Motion saliency

    `motion_saliency/get_motion_salient_boxes.m`  
        for each frame, prunes boxes (e.g. from Selective Search) based the optical flow signal within each box.  
           annot: set of videos and actions, jhmdb_annot.mat  
           ss_dir: directory containing the boxes  
           flow_dir: directory containing the optical flow images (as computed by A.)

5. Extract fc7 features

    `extract_features/rcnn_cache_fc7_features_jhmdb.m`  
        extracts fc7 features  
           type: 'spatial' or 'motion'  
           net_def_file: prototxts, `models/jhmdb/extract_fc7.prototxt`. Same for any type  
           net_file: models, use pretrained models for JHMDB as provided in the project page  
           output_dir: cache directory, the features are cached in `output_dir/type/action/video/frame.mat` 

6. Train SVM models

  `train_svm/train_jhmdb.m`  
     trains SVM models, one for each action  
        annot: ground truth information and boxes (after pruning), jhmdb_motion_sal_annot.mat  
        feat_dir: directory with cached features  
        save_dir: cache directory

7. Action Tubes

  `train_svm/compute_tubes.m`  
     scores and links detections to create the final action tubes  
       annot: source of boxes, `jhmdb_motion_sal_annot.mat`  
       rcnn_model: the models as computed by train_jhmdb.m

8. Precomputed tubes

  `test_tubes/`  
     tubes for all three splits of J-HMDB and UCF sports  
  `test_tubes/UCFsports_benchmark/`  
     AUC and ROC numbers for UCFSports and plots (see ipython notebook)

9. Evaluate/ROC curves

  `evaluate/get_ROC_curve_JHMDB.m`  
     computes ROC and AUC for JHMDB  
        annot: ground truth annotation (annot_jhdmb.mat)  
        tubes: tubes on the test set  
        actions: list of actions  
        iou_thresh: threshold for intersection over union  
        draw: true to draw the curves  
  (For UCF sports the same function was used with some small adjustments regarding the format of the data)


### Instructions

To train action tubes from scratch, you need to train two networks. 

Training spatial-CNN and motion-CNN
  To train the networks you need to do the following:
    1. Compute the optical flow as in A.    
    2. Create `window_train(val).txt` with the window data (similar to R-CNN detection)
    3. Use Caffe to train (train prototxt is given in `models/jhmdb/train.prototxt`), and initialize with the proper model
    4. In the case of motion-CNN, you need to make two changes in `window_data_layer.cpp`
      a. The image mean is for all channels 128 (instead of the image mean provided)
      b. During training, when flipping of the input image occurs the flow in x needs to also change sign

