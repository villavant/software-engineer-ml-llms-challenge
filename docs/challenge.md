Challenge Description

En este desafío, el objetivo principal fue transformar el notebook de exploración exploration.ipynb en un módulo de Python (model.py) respetando los parámetros definidos en el análisis inicial. Posteriormente, se integró la implementación del modelo en una API mediante api.py, exponiendo el servicio en el puerto 8080.

Pasos realizados:

Conversión de exploration.ipynb a model.py

Se refactorizó el código del notebook en un módulo reutilizable.

Se aseguraron los parámetros y la lógica del modelo para mantener coherencia con la exploración inicial.

Implementación en api.py

Se creó un endpoint para exponer el modelo.

Se utilizó el puerto 8080 para servir la API.

Se validó la correcta integración y funcionamiento del modelo a través de la API.

Modificación de cd.yml y ci.yml

Se realizaron ajustes en los archivos de integración y despliegue continuo según lo solicitado en la Parte 4.

Despliegue en GCP

Se presentaron complicaciones en la Parte 3 al desplegar en GCP, por lo que no se modificó la línea 26 del Makefile.

Este desafío permite consolidar buenas prácticas en la transición de exploración a producción, asegurando modularidad y escalabilidad en la implementación del modelo.