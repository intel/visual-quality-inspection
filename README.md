# Anomaly Detection: Visual Quality Inspection in the Industrial Domain

The goal of anomaly detection is to identify rare, abnormal events such as defects in a part being manufactured on an industrial production line. 


## **Table of Contents**
- [Overview](#overview)
- [Hardware Requirements](#hardware-requirements)
- [Software Requirements](#software-requirements)
- [How it Works?](#how-it-works)
- [Get Started](#get-started)
    - [Download the Workflow Repository](#Download-the-Workflow-Repository)
    - [Download the Transfer Learning Tool](#Download-the-Transfer-Learning-Tool)
    - [DataSet](#DataSet)
- [Ways to run this reference use case](#Ways-to-run-this-reference-use-case)
- [Run Using Docker](#run-using-docker)
- [Run Using Argo Workflows on K8s using Helm](#run-using-argo-workflows-on-k8s-using-helm)
- [Run Using Bare Metal](#run-using-bare-metal) 
- [Expected Output](#expected-output)
- [Summary and Next Steps](#summary-and-next-steps)
- [Learn More](#learn-more)
- [Support](#support)

## Overview
Classic and modern anomaly detection techniques have certain challenges: 
- Feature engineering needs to be performed to extract representations from the raw data. Traditional ML techniques rely on hand-crafted features that may not always generalize well to other settings. 
- Classification techniques require labeled training data, which is challenging because anomalies are typically rare occurrences and obtaining it increases the data collection & annotation effort. 
- Nature of anomalies can be arbitrary and unknown where failures or defects occur for a variety of unpredictable reasons, hence it may not be possible to predict the type of anomaly.

To overcome these challenges and achieve state-of-the-art performance, we present an unsupervised, mixed method end-to-end fine-tuning & inference reference solution for anomaly detection where a model of normality is learned from defect-free data in an unsupervised manner, and deviations from the models are flagged as anomalies. This reference use case is accelerated by Intel optimized software and is built upon easy-to-use Intel Transfer Learning Tool APIs.


## Hardware Requirements
There are workflow-specific hardware and software setup requirements depending on how the workflow is run. Bare metal development system and Docker image running locally have the same system requirements. 

| Recommended Hardware         | Precision  |
| ---------------------------- | ---------- |
| Intel® 4th Gen Xeon® Scalable Performance processors| FP32, BF16 |
| Intel® 1st, 2nd, 3rd, and 4th Gen Xeon® Scalable Performance processors| FP32 |


## Software Requirements 
Linux OS (Ubuntu 20.04) is used in this reference solution. Make sure the following dependencies are installed.

1. `sudo apt update`
1. python 3.9, pip/conda OR python3.9-venv
1. git

## How It Works?

This reference use case uses a deep learning based approach, named deep-feature modeling (DFM) and falls within the broader area of out-of-distribution (OOD) detection i.e. when a model sees an input that differs from its training data, it is marked as an anomaly. 

The use case provides 3 options for network modelling of the vision subtask:
* **Pre-trained backbone:** uses a deep network (ResNet-50v1.5 in this case) that has been pretrained on large visual datasets such as ImageNet
* **SimSiam self-supervised learning:** is a contrastive learning method based on Siamese networks. It learns meaningful representation of dataset without using any labels. SimSiam requires a dataloader such that it can produce two different augmented images from one underlying image. The end goal is to train the network to produce same features for both images. It takes a ResNet model as the backbone and fine-tunes the model on the augmented dataset to get closer feature embeddings for the use case. Read more [here](https://arxiv.org/pdf/2011.10566.pdf)
* **Cut-Paste self-supervised learning:** is a contrastive learning method similar to SimSiam but differs in the augmentations used during training. It take a ResNet model as backbone and fine-tunes the model after applying a data augmentation strategy that cuts an image patch and pastes at a random location of a large image. This allows us to construct a high performance model for defect detection without presence of anomalous data. Read more [here](https://arxiv.org/pdf/2104.04015.pdf)

![visual_quality_inspection_pipeline](assets/visual_quality_inspection_pipeline.JPG)
*Figure 1: Visual quality inspection pipeline. Above diagram is an example when using SimSiam self-supervised training*

Training stage only uses defect-free data. Images are loaded using a dataloader and shuffling, resizing & normalization processing is applied. Then one of the above stated transfer learning technique is used to fine-tune a model and extract discriminative features from an intermediate layer. A PCA kernel is trained over these features to reduce the dimension of the feature space while retaining 99% variance. This pre-processing of the intermediate features of a DNN is needed to prevent matrix singularities and rank deficiencies from arising.

During inference, the feature from a test image is generated through the same network as before. We then run a PCA transform using the trained PCA kernel and apply inverse transform to recreate original features and generate a feature-reconstruction error score, which is the norm of the difference between the original feature vector and the pre-image of its corresponding reduced embedding. Any image with an anomaly will have a high error in reconstructing original features due to features being out of distribution from the defect-free training set and will be marked as anomaly. The effectiveness of these scores in distinguishing the good images from the anomalous images is assessed by plotting the ROC curve, which is a plot of the true positive rate (TPR) of the classifier against the false positive rate (FPR) as the classification score-threshold is varied. The AUROC metric summarizes this curve between 0 to 1, with 1 indicating perfect classification.



**Architecture:**
![Visual_quality_inspection_layered_architecture](assets/Visual_quality_inspection_layered_architecture.JPG)

### Highlights of Visual Quality Inspection Reference Use Case
- The use case is presented in a modular architecture. To improve productivity and reduce time-to-solution, transfer learning methods are made available through an independent workflow that seamlessly uses Intel Transfer Learning Tool APIs underneath and a config file allows the user to change parameters and settings without having to deep-dive and modify the code.
- There is flexibility to select any pre-trained model and any intermediate layer for feature extraction.
- The use case is enabled with Intel optimized foundational tools


## Get Started
### Download the Workflow Repository
Create a working directory for the reference use case and clone the [Visual Quality Inspection Workflow](https://github.com/intel/visual-quality-inspection) repository into your working directory.
```
* Remove this line in final release *
git clone https://github.com/intel-innersource/frameworks.ai.end2end-ai-pipelines.anomaly-detection-ref-use-case.git
cd frameworks.ai.end2end-ai-pipelines.anomaly-detection-ref-use-case

git clone https://github.com/intel/visual-quality-inspection
cd visual-quality-inspection
```

### Download the Transfer Learning Tool (TLT)
```
* Remove this line in final release *
git clone https://github.com/intel-innersource/frameworks.ai.transfer-learning.git
git checkout pratool/anomaly_detection

git clone https://github.com/IntelAI/transfer-learning.git

```

### DataSet
[MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) is a dataset for benchmarking anomaly detection methods focused on visual quality inspection in the industrial domain. It contains over 5000 high-resolution images divided into ten unique objects and five unique texture categories. Each category comprises a set of defect-free training images and a test set of images with various kinds of defects as well as defect-free images. There are 73 different types of anomalies in the form of defects or structural deviations present in these objects and textures.

More information can be in the paper [MVTec AD – A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection](https://www.mvtec.com/fileadmin/Redaktion/mvtec.com/company/research/datasets/mvtec_ad.pdf)

![Statistical_overview_of_the_MVTec_AD_dataset](assets/mvtec_dataset_characteristics.JPG)

*Table 1:  Statistical overview of the MVTec AD dataset. For each category, the number of training and test images is given together with additional information about the defects present in the respective test images.[Source](https://www.mvtec.com/fileadmin/Redaktion/mvtec.com/company/research/datasets/mvtec_ad.pdf)*


#### Download the data

Download the mvtec dataset using Intel Model Zoo dataset download API
```
git clone https://github.com/intel-innersource/frameworks.ai.models.intel-models.git
cd frameworks.ai.models.intel-models/datasets/dataset_api/
pip install -r requirements.txt
python dataset.py -n mvtec-ad --download -d ../../../
```

Extract the tar file
```
cd ../../../
mkdir mvtec_dataset
tar -xf mvtec_anomaly_detection.tar.xz --directory mvtec_dataset
```



## Ways to run this reference use case
This reference kit offers three options for running the fine-tuning and inference processes:

- Docker
- Argo Workflows on K8s Using Helm
- Bare Metal

Details about each of these methods can be found below.

## Run Using Docker
Follow these instructions to set up and run our provided Docker image. For running on bare metal, see the [bare metal](#run-using-bare-metal) instructions.

### 1. Set Up Docker Engine and Docker Compose
You'll need to install Docker Engine on your development system. Note that while **Docker Engine** is free to use, **Docker Desktop** may require you to purchase a license. See the [Docker Engine Server installation instructions](https://docs.docker.com/engine/install/#server) for details.


To build and run this workload inside a Docker Container, ensure you have Docker Compose installed on your machine. If you don't have this tool installed, consult the official [Docker Compose installation documentation](https://docs.docker.com/compose/install/linux/#install-the-plugin-manually).


```bash
DOCKER_CONFIG=${DOCKER_CONFIG:-$HOME/.docker}
mkdir -p $DOCKER_CONFIG/cli-plugins
curl -SL https://github.com/docker/compose/releases/download/v2.7.0/docker-compose-linux-x86_64 -o $DOCKER_CONFIG/cli-plugins/docker-compose
chmod +x $DOCKER_CONFIG/cli-plugins/docker-compose
docker compose version
```

### 2. Install Vision Packages and Intel TensorFlow Toolkit
Ensure you have completed steps in the [Get Started Section](#get-started).

### 3. Set Up Docker Image
Build or Pull the provided docker image.

```bash
git submodule update --init --recursive
cd docker
docker compose build
```
OR
```bash
docker pull intel/ai-workflows:beta-anomaly-detection
docker pull intel/ai-workflows:beta-tlt-anomaly-detection
```

### 4. Preprocess Dataset with Docker Compose
Prepare dataset for Anomaly Detection workflows and accept the legal agreement to use the Intel Dataset Downloader.

```bash
cd docker
docker compose run preprocess -e USER_CONSENT=y
```

| Environment Variable Name | Default Value | Description |
| --- | --- | --- |
| DATASET_DIR | `$PWD/../data` | Unpreprocessed dataset directory |
| USER_CONSENT | n/a | Consent to legal agreement | <!-- TCE: Please help me word this better -->

### 5. Run Pipeline with Docker Compose

The Vision Finetuning container must complete successfully before the Evaluation container can begin. The Evaluation container uses the model and checkpoint files created by the vision fine-tuning container stored in the `${OUTPUT_DIR}` directory to complete the evaluation tasks.


```mermaid
%%{init: {'theme': 'dark'}}%%
flowchart RL
  VDATASETDIR{{"/${DATASET_DIR"}} x-. "-$PWD/../data}" .-x stocktltfinetuning
  VCONFIGDIR{{"/${CONFIG_DIR"}} x-. "-$PWD/../configs}" .-x stocktltfinetuning
  VOUTPUTDIR{{"/${OUTPUT_DIR"}} x-. "-$PWD/../output}" .-x stocktltfinetuning
  VDATASETDIR x-. "-$PWD/../data}" .-x stockevaluation
  VCONFIGDIR x-. "-$PWD/../configs}" .-x stockevaluation
  VOUTPUTDIR x-. "-$PWD/../output}" .-x stockevaluation
  stockevaluation --> stocktltfinetuning

  classDef volumes fill:#0f544e,stroke:#23968b
  class Vsimsiam,VDATASETDIR,VCONFIGDIR,VOUTPUTDIR,,VDATASETDIR,VCONFIGDIR,VOUTPUTDIR volumes
```

Run entire pipeline to view the logs of different running containers.

```bash
cd docker
docker compose run stock-evaluation &
```

| Environment Variable Name | Default Value | Description |
| --- | --- | --- |
| CONFIG | `eval` | Config file name |
| CONFIG_DIR | `$PWD/../configs` | Anomaly Detection Configurations directory |
| DATASET_DIR | `$PWD/../data` | Preprocessed dataset directory |
| OUTPUT_DIR | `$PWD/../output` | Logfile and Checkpoint output |

#### View Logs
Follow logs of each individual pipeline step using the commands below:

```bash
docker compose logs stock-tlt-fine-tuning -f
```

To view inference logs
```bash
fg
```

### 6. Run One Workflow with Docker Compose
Create your own script and run your changes inside of the container or run the evaluation without waiting for fine-tuning.

```mermaid
%%{init: {'theme': 'dark'}}%%
flowchart RL
  Vtransferlearning{{../transfer-learning}} x-.-x dev
  VCONFIGDIR{{"/${CONFIG_DIR"}} x-. "-$PWD/../configs}" .-x dev
  VDATASETDIR{{"/${DATASET_DIR"}} x-. "-$PWD/../data}" .-x dev
  VOUTPUTDIR{{"/${OUTPUT_DIR"}} x-. "-$PWD/output}" .-x dev

  classDef volumes fill:#0f544e,stroke:#23968b
  class Vtransferlearning,VCONFIGDIR,VDATASETDIR,VOUTPUTDIR volumes
```

Run using Docker Compose.

```bash
cd docker
docker compose run dev
```

| Environment Variable Name | Default Value | Description |
| --- | --- | --- |
| CONFIG | `eval` | Config file name |
| CONFIG_DIR | `$PWD/../configs` | Anomaly Detection Configurations directory |
| DATASET_DIR | `$PWD/../data` | Preprocessed Dataset |
| OUTPUT_DIR | `$PWD/output` | Logfile and Checkpoint output |
| SCRIPT | `anomaly_detection.py` | Name of Script |

#### Run Docker Image in an Interactive Environment

If your environment requires a proxy to access the internet, export your
development system's proxy settings to the docker environment:
```bash
export DOCKER_RUN_ENVS="-e ftp_proxy=${ftp_proxy} \
  -e FTP_PROXY=${FTP_PROXY} -e http_proxy=${http_proxy} \
  -e HTTP_PROXY=${HTTP_PROXY} -e https_proxy=${https_proxy} \
  -e HTTPS_PROXY=${HTTPS_PROXY} -e no_proxy=${no_proxy} \
  -e NO_PROXY=${NO_PROXY} -e socks_proxy=${socks_proxy} \
  -e SOCKS_PROXY=${SOCKS_PROXY}"
```

Run the workflow with the ``docker run`` command, as shown:

```bash
export CONFIG_DIR=$PWD/../configs
export DATASET_DIR=$PWD/../data
export OUTPUT_DIR=$PWD/../output
docker run -a stdout ${DOCKER_RUN_ENVS} \
           -e PYTHONPATH=/workspace/transfer-learning \
           -v /$PWD/../transfer-learning:/workspace/transfer-learning \
           -v /${CONFIG_DIR}:/workspace/configs \
           -v /${DATASET_DIR}:/workspace/data \
           -v /${OUTPUT_DIR}:/workspace/output \
           --privileged --init -it --rm --pull always --shm-size=8GB \
           intel/ai-workflows:beta-anomaly-detection \
           bash
```

Run the command below for fine-tuning and inference:
```bash
python /workspace/anomaly_detection.py --config_file /workspace/configs/finetuning.yaml
```

### 7. Clean Up Docker Containers
Stop containers created by docker compose and remove them.

```bash
docker compose down
```

## Run Using Argo Workflows on K8s Using Helm
### 1. Install Helm
- Install [Helm](https://helm.sh/docs/intro/install/)
```bash
curl -fsSL -o get_helm.sh https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 && \
chmod 700 get_helm.sh && \
./get_helm.sh
```
### 2. Setting up K8s
- Install [Argo Workflows](https://argoproj.github.io/argo-workflows/quick-start/) and [Argo CLI](https://github.com/argoproj/argo-workflows/releases)
- Configure your [Artifact Repository](https://argoproj.github.io/argo-workflows/configure-artifact-repository/)
- Ensure that your dataset and config files are present in your chosen artifact repository.
### 3. Install Workflow Template
```bash
export NAMESPACE=argo
helm install --namespace ${NAMESPACE} --set proxy=${http_proxy} anomaly-detection ./chart
argo submit --from wftmpl/workspace --namespace=${NAMESPACE}
```
### 4. View 
To view your workflow progress
```bash
argo logs @latest -f
```

## Run Using Bare Metal

### 1. Create environment and install software packages

Using conda:
```
conda create -n anomaly_det_refkit python=3.9
conda activate anomaly_det_refkit
pip install -r requirements.txt
```

Using virtualenv:
```
python3 -m venv anomaly_det_refkit
source anomaly_det_refkit/bin/activate
pip install -r requirements.txt
```

### 2. Select parameters and configurations

Select the parameters and configurations in the [finetuning.yaml](configs/README.md) file

NOTE: 
When using SimSiam self supervised training, download the Sim-Siam weights based on ResNet50 model and place under simsiam directory
```
wget https://dl.fbaipublicfiles.com/simsiam/models/100ep-256bs/pretrain/checkpoint_0099.pth.tar -o ./simsiam/checkpoint_0099.pth.tar
```

### 3. Running the end-to-end use case 

Using Transfer Learning Tool based fine-tuning:

In finetuning.yaml, change 'fine_tune' flag to true and set the simsiam/cutpaste settings accordingly
Change other settings in config.yaml to run different configurations
```
python anomaly_detection.py --config_file ./configs/finetuning.yaml
```

Using a pre-trained customized model:

In finetuning.yaml, change 'fine_tune' flag to false and provide a custom model path under 'saved_model_path'
Change other settings in config.yaml to run different configurations
```
python anomaly_detection.py --config_file ./configs/finetuning.yaml
```

## Expected Output

```
+------------+------------------------+-------+--------------+
|  Category  | Test set (Image count) | AUROC | Accuracy (%) |
+------------+------------------------+-------+--------------+
|   BOTTLE   |           83           | 100.0 |    100.0     |
|   CABLE    |          150           | 96.08 |    87.33     |
|  CAPSULE   |          132           | 92.62 |    96.21     |
|   CARPET   |          117           | 87.72 |    79.49     |
|    GRID    |           78           | 90.98 |    87.18     |
|  HAZELNUT  |          110           | 99.57 |    96.36     |
|  LEATHER   |          124           |  99.8 |    98.39     |
| METAL_NUT  |          115           | 92.38 |    86.96     |
|    PILL    |          167           | 96.92 |    91.02     |
|   SCREW    |          160           | 83.97 |    78.75     |
|    TILE    |          117           | 97.69 |    97.44     |
| TOOTHBRUSH |           42           | 97.78 |    95.24     |
| TRANSISTOR |          100           | 96.46 |     90.0     |
|    WOOD    |           79           | 97.72 |    93.67     |
|   ZIPPER   |          151           | 98.19 |    88.74     |
+------------+------------------------+-------+--------------+
```

## Summary and Next Steps


## Learn More
For more information or to read about other relevant workflow examples, see these guides and software resources:
- [Intel® AI Analytics Toolkit (AI Kit)](https://www.intel.com/content/www/us/en/developer/tools/oneapi/ai-analytics-toolkit.html)
- [Intel® Neural Compressor](https://github.com/intel/neural-compressor)
- [Intel® Extension for PyTorch](https://intel.github.io/intel-extension-for-pytorch/)
- [Intel® Transfer Learning Tool](https://github.com/IntelAI/models/tree/master/docs/notebooks/notebooks/e2e_workflows/Anomaly_Detection_MVTec.ipynb)

## Support
If you have any questions with this workflow, want help with troubleshooting, want to report a bug or submit enhancement requests, please submit a GitHub issue
