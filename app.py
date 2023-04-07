import torch
import gradio as gr
from PIL import Image
from model import get_model, apply_weights, copy_weight
from vocab import vocab
from transforms import resized_crop_pad, gpu_crop
from torchvision.transforms import Normalize, ToTensor

model = get_model()
state = torch.load("./vit_saved.pth", map_location="cpu")
apply_weights(model, state, copy_weight)

to_tensor = ToTensor()
norm = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def decode_pred(pred: torch.Tensor) -> str:
    indices = pred > 0.95
    if indices.any():
        # return first match
        return vocab[indices.nonzero()[0]]
    else:
        return "I don't know what this is, ¡páharo!"


def classify_image(inp):
    inp = Image.fromarray(inp)
    transformed_input = resized_crop_pad(inp, (460, 460))
    transformed_input = to_tensor(transformed_input).unsqueeze(0)
    transformed_input = gpu_crop(transformed_input, (224, 224))
    transformed_input = norm(transformed_input)
    model.eval()
    with torch.no_grad():
        pred = model(transformed_input)
    return decode_pred(torch.sigmoid(pred).squeeze(dim=0))


iface = gr.Interface(
    fn=classify_image,
    inputs=gr.inputs.Image(),
    outputs="text",
    title="Birds Classifier without Fastai",
    description="A birds classifier over 200 species trained with Fastai"
    " and deployed with plain pytorch in Gradio.",
).launch()
