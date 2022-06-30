import pytorch_lightning as pl
import timm


class ImageBackbone(pl.LightningModule):
    def __init__(self, model_name, trunk=None, **kwargs):
        super().__init__()
        self.model_name = model_name
        self.trunk = trunk
        if trunk is None:
            self.trunk = timm.create_model(
                self.model_name,
                pretrained=False,
                num_classes=0,
            )
        self.save_hyperparameters()

    def forward(self, images):
        embeddings = self.trunk(images)
        return embeddings
