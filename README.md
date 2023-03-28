# Mt. Whitney Reference Use Case: Anomaly Detection

### 1. Clone the Anomaly Detection reference use case repository
   ```
   git clone https://github.com/intel-innersource/frameworks.ai.end2end-ai-pipelines.anomaly-detection-ref-use-case.git
   cd frameworks.ai.end2end-ai-pipelines.anomaly-detection-ref-use-case
   ```

### 2. Create ~~conda~~ pip venv and install software packages
   ```
   #conda create -n anomaly_det_refkit python=3.9
   #conda activate anomaly_det_refkit
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

### 3. Download the Transfer Learning Toolkit (TLT)
   ```
   git clone https://github.com/intel-innersource/frameworks.ai.transfer-learning.git
   ```

### 4. Download and prepare the dataset

   Download the MVTEC dataset from: https://www.mvtec.com/company/research/datasets/mvtec-ad/downloads

   ```
   wget https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938113-1629952094/mvtec_anomaly_detection.tar.xz
   ```

   Extract 'mvtec_anomaly_detection.tar.xz' using following commands:
   ```
   mkdir -p mvtec
   tar -xf mvtec_anomaly_detection.tar.xz --directory mvtec
   ```

   Generate the CSV file for each category of MVTEC dataset using following command. It will automatically place the CSV files under each category directory:
   ```
   python csv_generator_mvtec.py --path /path/to/mvtec/
   ```

### 5. Setup the feature extractor

   We have three feature extractor options.
   ```
   First - SimSiam - A self-supervised method that takes ResNet50 model as backbone and fine-tune the model on custom dataset to get better feature embedding
   ```
   Download the Sim-Siam weights based on ResNet50 model and place under simsiam directory
   ```
   wget https://dl.fbaipublicfiles.com/simsiam/models/100ep-256bs/pretrain/checkpoint_0099.pth.tar -o ./simsiam/checkpoint_0099.pth.tar
   ```
   ```
   Second - Cut-paste - A self-supervised method that takes ResNet50/ ResNet18 model as backbone and fine-tune the model on custom dataset to get better feature embedding
   ```
   ```
   Third - No fine-tuning and just use pretrained ResNet50/ResNet18 model for feature extraction
   ```
### 6. Running the workload

   ```
   python anomaly-detection.py

   Optional arguments:
     -h, --help           show this help message and exit  
     --simsiam            flag to fine-tune simsiam feature extractor  
     --cutpaste           flag to fine-tune cutpaste feature extractor
     --freeze_resnet      number of epochs until cutpaste backbone ResNet layers will be frozen and only head layers will be trained
     --cutpaste_type      options for cutpaste augmentations {normal,scar,3way,union - default is normal} 
     --head_layer         number of fully connected layers following ResNet backbone in cutpaste
     --epochs EPOCHS      number of epochs to train SimSiam feature extractor  
     --data PATH          path for MVTEC base dataset directory, i.e. /path/to/mvtec/  
     --category CATEGORY  category of the dataset, i.e. hazelnut or all  
   ```



