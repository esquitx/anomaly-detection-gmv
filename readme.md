# Modelización de problemas de empresas UCM . GMV. "Anomaly detection in MVTec AD"


For this project, information was retrieved from the following sources: 

[1] Intro to Autoencoders. TensorFlow Core Tutorials. Retrieved from https://www.tensorflow.org/tutorials/generative/autoencoder 

[2] Sarafijanovic-Djukic N, Davis J. Fast Distance-Based Anomaly Detection in Images Using an Inception-Like Autoencoder. InInternational Conference on Discovery Science. 2019 Oct 28. (pp. 493-508). Springer, Cham. 

[3] AdneneBoumessouer. MVTec-Anomaly-Detection. Retrieved from https://github.com/AdneneBoumessouer/MVTec-Anomaly-Detection. Última revisión 2021 18 Nov. 

[4] natasasdj. Anomaly Detection. Retrieved from https://github.com/natasasdj/anomalyDetection. Última revisión 2021 18. Nov. 

[5] Ehret, T., Davy, A., Morel, J., & Delbracio, M. (2019, June 03). Image Anomalies: A Review and Synthesis of Detection Methods. Retrieved from https://arxiv.org/abs/1808.02564 

[6] Aditya Sharma (2018, Abril 4) Implementing autoencoders in Keras : Tutorial. Retrieved from https://www.datacamp.com/community/tutorials/autoencoder-keras-tutorial

[7] Redes Neuronales Residuales - Lo que necesitas saber (ResNet). (2019) Retrieved from https://datascience.eu/es/aprendizaje-automatico/una-vision-general-de-resnet-y-sus -variantes/ 

[8] Mariano Rivera. La Red Residual (Residual Network, ResNet). (2019, Agosto) Retrieved from http://personal.cimat.mx:8181/~mrivera/cursos/aprendizaje_profundo/resnet/resnet.ht ml 

Most of the project was inspired in the the AdneneBoumessouer/MVTec-Anomaly-Detection repo in [3].

## Sample use

### train.py
```
python3 train.py -d (dir_datos) -t normal -l (mssim)
```

### test.py
```
python3 test.py -d modelos_guardados/normal/mssim/18-11-2021-21-13-50
```

