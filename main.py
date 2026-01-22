
import torch
from pathlib import Path
from anomalib.data import MVTec
from anomalib.models import EfficientAD
from anomalib.engine import Engine


def main():

    print("Version de pytorch: ", torch.__version__)
    print("Grafica disponible?: {'Si' if torch.cuda.is_available() else 'No'}", )
    print("Version de cuda: ", torch.version.cuda)
    if torch.cuda.is_available(): 
        print("Nombre de grafica: ", torch.cuda.get_device_name(0))


    data_root = Path(r"./data/mvtec_anomaly_detection/")
    category = "bottle"
    category_img_res = (900,900)

    # 1) DataModule: MVTec (ya calza con tu estructura train/test/ground_truth)
    datamodule = MVTec(
        root=data_root,
        category=category,
        image_size=category_img_res,       # común para empezar; luego lo ajustas
        train_batch_size=16,
        eval_batch_size=16,
        num_workers=8,
        task="segmentation",         # porque tienes ground_truth masks
    )

    model = EfficientAD(
        image_size=category_img_res,
    )

    engine = Engine(
        task="segmentation",
        accelerator="auto",  # usa GPU si está disponible
        devices=1,
    )

    # Entrenar
    engine.fit(datamodule=datamodule, model=model)

    # Evaluar en test (usa test/* y ground_truth/* para métricas pixel-level)
    engine.test(datamodule=datamodule, model=model)

if __name__ == "__main__":
    main()
