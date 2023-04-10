# Mt. Whitney Reference Use Case: Anomaly Detection

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

### 5. Feature Extractor

   We have three feature extractor options.
   ```
   First - SimSiam - A self-supervised method that takes ResNet50 model as backbone and fine-tune the model on custom dataset to get better feature embedding
   ```
   Download the Sim-Siam weights based on ResNet50 model and place under simsiam directory
   ```
   https://dl.fbaipublicfiles.com/simsiam/models/100ep-256bs/pretrain/checkpoint_0099.pth.tar
   mv checkpoint_0099.pth.tar simsiam
   ```
   ```
   Second - Cut-paste - A self-supervised method that takes ResNet50/ ResNet18 model as backbone and fine-tune the model on custom dataset to get better feature embedding
   ```
   ```
   Third - Pretrained - No fine-tuning and just use pretrained ResNet50/ResNet18 model for feature extraction
   ```
### 6. Running the workload

   ```
   python anomaly_detection.py --config config.yaml

   Change settings in config.yaml to run different configurations 

   ```



