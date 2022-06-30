FROM mcr.microsoft.com/azureml/pytorch-1.9-ubuntu18.04-py37-cuda11.0.3-gpu-inference:20220516.v3

USER root:root

RUN pip install --upgrade pip
RUN pip install azureml==0.2.7 azureml-core==1.42.0 azureml-mlflow==1.42.0

RUN pip install setuptools==59.5.0 \
                ipython==7.32.0 \
                torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html \
                torchtext==0.11.0 \
                pytorch_metric_learning==1.1.0 \
                pytorch-lightning==1.6.1 \
                transformers==4.17.0 \
                fasttext==0.9.2 \
                pandas==1.3.5 \
                mlflow==1.26.1 \
                scikit-learn==1.0.2 \
                statsmodels==0.13.2 \
                crowd-kit==1.0.0 \
                optuna==2.10.0 \
                timm==0.5.4 \
                torchmetrics==0.9.1 \
                wandb==0.12.18 \
                plotly matplotlib jupyterlab ipywidgets>=7.6 jupyter-dash umap-learn --upgrade

CMD ["/bin/bash"]
