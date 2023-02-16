# frameworks.ai.end2end-ai-pipelines.anomaly-detection-ref-use-case

## To clone the repository
1. git clone https://github.com/intel-innersource/frameworks.ai.end2end-ai-pipelines.anomaly-detection-ref-use-case.git
2. cd frameworks.ai.end2end-ai-pipelines.anomaly-detection-ref-use-case

## To prepare the MVTEC dataset ---

1. Download the MVTEC 'whole' dataset from the following link 
   https://www.mvtec.com/company/research/datasets/mvtec-ad/downloads
2. Extract 'mvtec_anomaly_detection.tar.xz' using following command
   tar -xf mvtec_anomaly_detection.tar.xz
3. Generate the CSV file for each category of MVTEC dataset using following command. It will automatically place the CSV files under each category directory
   python csv_generator_mvtec.py --path /path/to/mvtec/

## To setup the SimSiam feature extractor ---

1. Download the Sim-Siam weights based on ResNet50 model from the link and move it inside ./simsiam directory-
   https://dl.fbaipublicfiles.com/simsiam/models/100ep-256bs/pretrain/checkpoint_0099.pth.tar


## To execute the code ---

1. Provide the MVTEC dataset directory path
2. Execute "python anomaly-detection.py" command

Optional arguments:
  -h, --help           show this help message and exit
  --simsiam            flag to enable simsiam feature extractor
  --epochs EPOCHS      number of epochs to train SimSiam feature extractor
  --path PATH          path for MVTEC base dataset directory, i.e. /path/to/mvtec/
  --category CATEGORY  category of the dataset, i.e. hazelnut or all

3. To enable SimSiam feature extractor, use --simsiam flag. Deafult is imagenet trained ResNet50 feature extractor
4. To change the dataset, pass category variable to --category. To run all category at once, pass --category all
5. If SimSiam feature extractor is selected, pass --epochs, number of epochs for fine-tuning.



## To run the demo ---
1. Add hazelnut_demo.csv in hazelnut directory
2. Execute "Python anomaly_detection_demo.py"




