"""
module for ssl handlers
"""
import logging
import torch
from torchvision import transforms
from ts.torch_handler.vision_handler import VisionHandler
#from ..trainer.inputs import base_transform
from torchvision import transforms
logger = logging.getLogger(__name__)

def base_transform():
    return [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]

class SSLVision(VisionHandler):
    """
    SSLVision handler class. This handler extends base vision class
    This handler takes and image and returns the embedded vector for downstream vector db
    """

    image_processing = transforms.Compose(base_transform())

    def __init__(self):
        super(SSLVision, self).__init__()

    # def preprocess(self, data):
    #     images = []
    #     logger.info('raw', data)
    #     for row in data:
    #         logger.info(row)
    #         load_bytes = BytesIO(row.get('body'))
    #         #image = np.loadtxt()
    #         results = np.asarray(pd.read_json(load_bytes,orient='records').values.tolist()[0], dtype=np.uint8)
    #         logger.info(results.shape)
    #         #image = np.array(row.get("data") or row.get("body"))
    #         #logger.info('image',image)
    #         image = self.image_processing(results)
    #         images.append(image)
    #     return torch.stack(images).to(self.device)

    def inference(self, data, *args, **kwargs):
        marshalled_data = data.to(self.device)
        with torch.no_grad():
            results = torch.nn.functional.normalize(self.model.encoder(data, *args, **kwargs))
        return results
