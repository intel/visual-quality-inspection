# Setting parameters and configurations for the visual quality inspection use case

Please set the following in the finetuning.yaml file:

* **num_workers:** number of sub-processes or threads to use for data loading. Setting the argument num_workers as a positive integer will turn on multi-process data loading. (Default=32)

* **fine_tune:** set 'True' to run SimSiam or CutPaste self-supervised learning using Intel Transfer Learning Tool APIs. Set 'False' to run a pre-trained backbone by providing a model path under 'saved_model_path' category

* **output_path:** path to save the checkpoints or final model

* **tlt_wf_path:** set by default to point to the workflow in the Intel Transfer Learning Tool

* **dataset:**
  * **root_dir:** path to the root directory of MVTEC dataset
  * **category_type:** category type within MVTEC dataset, e.g.: hazelnut or all (for running all categories in MVTEC)
  * **batch_size:** batch size for inference (Default=32)
  * **image_size:** each image resized to this size (Default=224x224)

* **model:** Options to select when running with a pre-trained backbone, no fine-tuning on custom dataset
  * **name:** pretrained backbone model E.g.: resnet50, resnet18
  * **layer:** intermediate layer from which features will be extracted
  * **pool:** pooling kernel size for average pooling
  * **feature_extractor:** select the type of modelling and subsequent feature extractor. Options are:
    * pretrained -  No fine-tuning on custom dataset, features will be extracted from pretrained model which is set in model/name
    * simsiam - SimSiam self-supervised training on custom dataset
    * cutpaste - CutPaste self-supervised training on custom dataset 

* **simsiam:** Set when 'feature_extractor' is set to simsiam.
  * **batch_size:** batch size for fine-tuning (Default=64)
  * **epochs:** number of epochs to fine-tune the model
  * **optim:** optimization algorithm E.g.: sgd, adam
  * **model_path:** path to save the checkpoints or final model
  * **ckpt:** flag to specify whether intermediate checkpoints should be saved or not

* **cutpaste:** Set when 'feature_extractor' is set to cutpaste.
  * **cutpaste_type:**
  * **head_layer:**
  * **freeze_resnet:**
  * **batch_size:** batch size for fine-tuning (Default=64)
  * **epochs:** number of epochs to fine-tune the model
  * **optim:** optimization algorithm E.g.: sgd, adam
  * **model_path:** path to save the checkpoints or final model
  * **ckpt:** flag to specify whether intermediate checkpoints should be saved or not

* **pca_thresholds:** percentage of variance ratio to be retained. Number of PCA components are selected according to it

