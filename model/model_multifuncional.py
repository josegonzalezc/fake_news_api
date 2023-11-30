import torch
import torch.nn as nn
from fastapi import File, UploadFile
from pydantic import BaseModel
from transformers import CLIPModel
from transformers import CLIPProcessor


class ClassificationModel(nn.Module):
    def __init__(self, pretrained_model="openai/clip-vit-base-patch32"):
        super(ClassificationModel, self).__init__()
        self.clip = CLIPModel.from_pretrained(pretrained_model)
        self.bilayer = nn.Bilinear(512, 512, 512)
        self.relu1 = nn.ReLU()
        self.linear1 = nn.Linear(512, 512)
        self.relu2 = nn.ReLU()
        self.linear2 = nn.Linear(512, 1)

    def forward(self, input_ids, attention_mask, pixel_values):
        clip_layer = self.clip(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values)
        x = self.bilayer(clip_layer.text_embeds, clip_layer.image_embeds)
        x = self.relu1(x)
        x = self.linear1(x)
        x = self.relu2(x)
        return self.linear2(x)

    def clip_freeze(self):
        model_weight = self.clip.state_dict().keys()
        model_weight_list = [*model_weight]
        for name, param in self.clip.named_parameters():
            if name in model_weight_list:
                param.requires_grad = False


def load_model(model_path='modelo/model1.pt', map_location=torch.device('cpu')):
    model = ClassificationModel()
    try:
        model.load_state_dict(torch.load(model_path, map_location=map_location))
        model.eval()  # Poner el modelo en modo de evaluación
    except Exception as e:
        print(f"Error loading the model: {e}")
    return model

def context_padding(inputs, context_length=77):
    input_length = inputs.input_ids.shape[1]
    if input_length > context_length:
        # Clip the input sequence to the context length
        input_ids = inputs.input_ids[:, :context_length]
        attention_mask = inputs.attention_mask[:, :context_length]
    else:
        # Pad the input sequence with zeros
        shape = (1, context_length - input_length)
        x = torch.zeros(shape)
        input_ids = torch.cat([inputs.input_ids, x], dim=1).long()
        attention_mask = torch.cat([inputs.attention_mask, x], dim=1).long()

    return input_ids, attention_mask


def get_prediction_string(prediction):
    """ Convierte la predicción en una cadena de texto representativa """
    return "Fake News" if prediction <= 0.5 else "Not A Fake News"

# Cargar el procesador CLIP
def load_processor():
    return CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

class Item(BaseModel):
    text: str
    image: UploadFile = File(...)


def predice(text, processed_image):
    print("================ PREDICT =========")
    inputs = processor(text=text, images=processed_image, return_tensors="pt", padding=True)
    input_ids, attention_mask = context_padding(inputs)

    # Realiza la predicción
    with torch.no_grad():
        predictions = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=inputs.pixel_values)

    # Convierte las predicciones a una respuesta JSON serializable
    predictions = predictions.cpu().numpy().tolist()  # Asume que el modelo devuelve un array de NumPy
    print(f"PREDICCION {predictions}")
    logits = torch.tensor(predictions)
    probabilities = torch.sigmoid(logits)
    print("probabilities.item()", probabilities.item())
    resultado = get_prediction_string(probabilities.item())
    return resultado


# Carga el modelo y el procesador previamente guardados
model = ClassificationModel()
model.load_state_dict(torch.load('modelo/model1.pt', map_location=torch.device('cpu')))
model.eval()  # Poner el modelo en modo de evaluación

processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
