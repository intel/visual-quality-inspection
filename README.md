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

### 5. Setup the SimSiam feature extractor

   Download the Sim-Siam weights based on ResNet50 model
   ```
   https://dl.fbaipublicfiles.com/simsiam/models/100ep-256bs/pretrain/checkpoint_0099.pth.tar
   mv checkpoint_0099.pth.tar simsiam
   ```


### 6. Running the workload

   ```
   python anomaly-detection.py

   Optional arguments:
     -h, --help           show this help message and exit  
     --simsiam            flag to enable simsiam feature extractor  
     --epochs EPOCHS      number of epochs to train SimSiam feature extractor  
     --path PATH          path for MVTEC base dataset directory, i.e. /path/to/mvtec/  
     --category CATEGORY  category of the dataset, i.e. hazelnut or all  
   ```

### 7. Run a demo

   1. Add hazelnut_demo.csv in hazelnut directory  
   2. Run the following command:
   ```
   Python anomaly_detection_demo.py
   ```





