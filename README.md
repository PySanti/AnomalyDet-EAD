# AnomalyDet-EAD

El objetivo de este proyecto es hacer pruebas con el modelo `EfficientAD` sobre el dataset `MV Tec`.

Se utilizara la libreria `anomalib` y su implementacion de `EfficientAD`

# Guia de instalacion



1. Clonacion de repositorio

*En la terminal*

`
git clone https://github.com/PySanti/AnomalyDet-EAD
`

2. Cambio de carpeta

`
cd AnomalyDet-EAD
`

3. Creacion de entorno virtual : es necesario utilizar `python3.10`

`
py -3.10 -m venv dep
`

4. Activacion de entorno virtual.

`
./dep/Scripts/activate
`

5. Instalacion de dependencias

`
pip3 install -r dep-list.txt
`

6. Ejecucion de pruebas

`
python3 main.py
`


# Estructura de proyecto utilizada

```

.
+---data
|   +---bottle
|   |   +---ground_truth
|   |   +---test
|   |   +---train
|   +---cable
|   |   +---ground_truth
|   |   +---test
|   |   +---train
|   +---capsule
|   |   +---ground_truth
|   |   +---test
|   |   +---train
|   +---carpet
|   |   +---ground_truth
|   |   +---test
|   |   +---train
|   +---grid
|   |   +---ground_truth
|   |   +---test
|   |   +---train
|   +---hazelnut
|   |   +---ground_truth
|   |   +---test
|   |   +---train
|   +---leather
|   |   +---ground_truth
|   |   +---test
|   |   +---train
|   +---metal_nut
|   |   +---ground_truth
|   |   +---test
|   |   +---train
|   +---pill
|   |   +---ground_truth
|   |   +---test
|   |   +---train
|   +---screw
|   |   +---ground_truth
|   |   +---test
|   |   +---train
|   +---tile
|   |   +---ground_truth
|   |   +---test
|   |   +---train
|   +---toothbrush
|   |   +---ground_truth
|   |   +---test
|   |   +---train
|   +---transistor
|   |   +---ground_truth
|   |   +---test
|   |   +---train
|   +---wood
|   |   +---ground_truth
|   |   +---test
|   |   +---train
|   +---zipper
|       +---ground_truth
|       +---test
|       +---train
+---dep
+---main.py
```

# Versiones utilizadas

```
python=3.10


aiohappyeyeballs==2.6.1
aiohttp==3.13.3
aiosignal==1.4.0
anomalib==2.2.0
antlr4-python3-runtime==4.9.3
anyio==4.12.1
async-timeout==5.0.1
attrs==25.4.0
build==1.4.0
certifi==2026.1.4
charset-normalizer==3.4.4
click==8.3.1
colorama==0.4.6
contourpy==1.3.2
cycler==0.12.1
docstring_parser==0.17.0
einops==0.8.1
exceptiongroup==1.3.1
filelock==3.20.3
fonttools==4.61.1
FrEIA==0.2
frozenlist==1.8.0
fsspec==2026.1.0
h11==0.16.0
hf-xet==1.2.0
httpcore==1.0.9
httpx==0.28.1
huggingface_hub==1.3.3
idna==3.11
imagecodecs==2025.3.30
ImageIO==2.37.2
importlib_metadata==8.7.1
importlib_resources==6.5.2
Jinja2==3.1.6
joblib==1.5.3
jsonargparse==4.45.0
kiwisolver==1.4.9
kornia==0.8.2
kornia_rs==0.1.10
lazy_loader==0.4
lightning==2.6.0
lightning-utilities==0.15.2
markdown-it-py==4.0.0
MarkupSafe==3.0.3
matplotlib==3.10.8
mdurl==0.1.2
mpmath==1.3.0
multidict==6.7.0
networkx==3.4.2
numpy==2.2.6
omegaconf==2.3.0
opencv-python==4.13.0.90
packaging==26.0
pandas==2.3.3
pillow==12.1.0
pip-tools==7.5.2
propcache==0.4.1
Pygments==2.19.2
pyparsing==3.3.2
pyproject_hooks==1.2.0
python-dateutil==2.9.0.post0
pytorch-lightning==2.6.0
pytz==2025.2
PyYAML==6.0.3
requests==2.32.5
rich==14.2.0
rich-argparse==1.7.2
safetensors==0.7.0
scikit-image==0.25.2
scikit-learn==1.7.2
scipy==1.15.3
shellingham==1.5.4
six==1.17.0
sympy==1.14.0
threadpoolctl==3.6.0
tifffile==2025.5.10
timm==1.0.24
tomli==2.4.0
torch==2.10.0+cu130
torchmetrics==1.8.2
torchvision==0.25.0+cu130
tqdm==4.67.1
typer-slim==0.21.1
typeshed_client==2.8.2
typing_extensions==4.15.0
tzdata==2025.3
urllib3==2.6.3
yarl==1.22.0
zipp==3.23.0
```

# Consideraciones

Correr estas pruebas en local es lento por que `anomalib` es una libreria construida sobre `pytorch`, esto hace que implemente internamente la clase DataLoader, para que que el entrenamiento sea mas rapido, el dataloader debe tener su atributo `persistent_workers=True` cosa que no se hace por que hace el entrenamiento inestable.

# Resultados

