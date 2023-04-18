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



