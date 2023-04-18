# Anomaly Detection

![Version: 0.1.0](https://img.shields.io/badge/Version-0.1.0-informational?style=flat-square) ![Type: application](https://img.shields.io/badge/Type-application-informational?style=flat-square) ![AppVersion: 1.16.0](https://img.shields.io/badge/AppVersion-1.16.0-informational?style=flat-square)

A Helm chart for Kubernetes

## Values

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| dataset.nfs.path | string | `"nil"` | Path to Local NFS Share in Cluster Host |
| dataset.nfs.server | string | `"nil"` | Hostname of NFS Server |
| dataset.nfs.subPath | string | `"nil"` | Path to dataset in Local NFS |
| dataset.s3.key | string | `"nil"` | Path to Dataset in S3 Bucket |
| dataset.type | string | `"<nfs/s3>"` | `nfs` or `s3` dataset input enabler |
| image.base | string | `"intel/ai-workflows"` | base container repository |
| image.tlt | string | `"beta-tlt-anomaly-detection"` | vision tlt workflow container tag |
| image.use_case | string | `"beta-anomaly-detection"` | evaluation container tag |
| metadata.name | string | `"anomaly-detection"` |  |
| proxy | string | `"nil"` |  |
| workflow.config.inference | string | `"eval"` | config file name for inference |
| workflow.config.training | string | `"finetuning"` | config file name for training |
