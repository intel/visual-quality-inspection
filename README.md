# Mt. Whitney Reference Use Case: Anomaly Detection

## Hardware Requirements
There are workflow-specific hardware and software setup requirements depending on how the workflow is run. Bare metal development system and Docker image running locally have the same system requirements. 

| Recommended Hardware         | Precision  |
| ---------------------------- | ---------- |
| Intel® 4th Gen Xeon® Scalable Performance processors| FP32, BF16 |
| Intel® 1st, 2nd, 3rd, and 4th Gen Xeon® Scalable Performance processors| FP32 |


### 1. Create conda env and install software packages
   ```
   conda create -n anomaly_det_refkit python=3.9
   conda activate anomaly_det_refkit
   pip install -r requirements.txt
   ```

### 2. Clone the Anomaly Detection reference use case repository
   ```
   git clone https://github.com/intel-innersource/frameworks.ai.end2end-ai-pipelines.anomaly-detection-ref-use-case.git
   cd frameworks.ai.end2end-ai-pipelines.anomaly-detection-ref-use-case
   ```

### 3. Download the Transfer Learning Toolkit (TLT)
   ```
   git clone https://github.com/intel-innersource/frameworks.ai.transfer-learning.git
   git checkout pratool/anomaly_detection
   ```

### 4. Download and prepare the dataset

   Download the MVTEC dataset from: https://www.mvtec.com/company/research/datasets/mvtec-ad/downloads

   Extract 'mvtec_anomaly_detection.tar.xz' using following command:
   ```
   tar -xf mvtec_anomaly_detection.tar.xz
   ```

   Generate the CSV file for each category of MVTEC dataset using following command. It will automatically place the CSV files under each category directory:
   ```
   python csv_generator_mvtec.py --path /path/to/mvtec/
   ```

### OR
   Download and pre-process the dataset using Model Zoo dataset download API
   ```
   git clone https://github.com/intel-innersource/frameworks.ai.models.intel-models.git
   git checkout wafaa/datasetapi-mvtec-dataset
   python dataset.py -n mvtec-ad --download --preprocess -d /data/datad/ad_best_testing/mvtec_dataset
   ```
   

### 5. Feature Extractor

   We have three feature extractor options.
   ```
   First - SimSiam - A self-supervised method that takes ResNet50 model as backbone and fine-tune the model on custom dataset to get better feature embedding
   ```
   Download the Sim-Siam weights based on ResNet50 model and place under simsiam directory
   ```
   wget https://dl.fbaipublicfiles.com/simsiam/models/100ep-256bs/pretrain/checkpoint_0099.pth.tar
   ```
   ```
   Second - Cut-paste - A self-supervised method that takes ResNet50/ ResNet18 model as backbone and fine-tune the model on custom dataset to get better feature embedding
   ```
   ```
   Third - Pretrained - No fine-tuning and just use pretrained ResNet50/ResNet18 model for feature extraction
   ```
### 6. Running the workload without TLT

   ```
   In config.yaml, change 'fine_tune' flag to false and provide model path under 'saved_model_path'  

   python anomaly_detection.py --config config.yaml

   Change other settings in config.yaml to run different configurations 

   ```

### 7. Running the workload with TLT

   ```
   In config.yaml, change 'fine_tune' flag to true and set the simsiam/cutpaste settings accordingly

   python anomaly_detection.py --config config.yaml

   Change other settings in config.yaml to run different configurations

   ```

