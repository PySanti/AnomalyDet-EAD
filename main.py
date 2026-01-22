import torch
from pathlib import Path
from anomalib.data import Folder, MVTecAD, AnomalibDataModule
from anomalib.models import EfficientAd
from anomalib.engine import Engine

if __name__ == "__main__":

    print("Version de pytorch: ", torch.__version__)
    print(f"Grafica disponible?: {'Si' if torch.cuda.is_available() else 'No'}", )
    print("Version de cuda: ", torch.version.cuda)
    if torch.cuda.is_available(): 
        print("Nombre de grafica: ", torch.cuda.get_device_name(0))
        if '5070' in torch.cuda.get_device_name(0):
            torch.set_float32_matmul_precision('high')


    data_root = Path(r"./data/")
    category = "bottle"
    datamodule = MVTecAD(
        root=data_root,
        category=category,
        train_batch_size=1,  # recomendado para empezar con EfficientAd
        eval_batch_size=1,
        num_workers=1,
    )

    model = EfficientAd()

    engine = Engine(
        accelerator="auto",
        devices=1,
        max_epochs=50
    )

    engine.fit(datamodule=datamodule, model=model)
    engine.test(datamodule=datamodule, model=model)


