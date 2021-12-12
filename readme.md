# MLOPS PROJECT

Idea of this project is to have simple cat vs dog classification project with emphasize on MLOps pipeline. Goal is to
have everything running on AWS.

### MLFlow for model versioning

1. `pip install mlflow`
2. `mlflow ui`
3. Go to http://localhost:5000

### LakeFS for data versioning

1. Follow these steps in order to set up
   lakeFS https://towardsdatascience.com/data-versioning-all-you-need-to-know-7077aa5ed6d1

### Running pipeline

1. `pip install requirements`
2. Create `config.ini` with your credentials. Example is in `config.ini.EXAMPLE`
3. Prepare data and put it under `dataset_cat_vs_dog` folder to have sub folders dog and cat with images.
4. `python train.py`
5. `export PYTHONPATH=$PYTHONPATH:$pwd`
6. `python prepare_and_export/export.py`
7. `docker run -t --rm -p 8501:8501 -v "PATH_TO_PROJECT/MLOps/serving:/models/dog_detector"     
   -e MODEL_NAME=dog_detector tensorflow/serving &`
8. `python prepare_and_export/inference.py`
