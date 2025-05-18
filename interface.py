# import streamlit as st
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import timm
# from PIL import Image
# import torchvision.transforms as transforms
# import numpy as np
# import cv2

# # --- Device configuration ---
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # --- Class labels ---
# CLASS_NAMES = ["Mild Demented", "Moderate Demented", "Non Demented", "Very Mild Demented"]

# # --- Vision Transformer Encoder ---
# class ViTEncoder(nn.Module):
#     def __init__(self, model_name='tf_efficientnetv2_b0', embed_dim=1280):
#         super().__init__()
#         self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0)
#         self.embed_dim = embed_dim

#     def forward(self, x):
#         return self.backbone(x)

# # --- Classifier Head ---
# class ClassifierHead(nn.Module):
#     def __init__(self, encoder, num_classes=4):
#         super().__init__()
#         self.encoder = encoder
#         self.classifier = nn.Linear(encoder.embed_dim, num_classes)

#     def forward(self, x):
#         features = self.encoder(x)
#         logits = self.classifier(features)
#         return logits

# # --- Model loading ---
# MODEL_DIR = "C:/Users/arfao/Desktop/imp_4c_dataset/models"

# @st.cache_resource(show_spinner=False)
# def load_models():
#     encoder = ViTEncoder().to(device)
#     checkpoint_ijepa = torch.load(f"{MODEL_DIR}/best_ijepa.pth", map_location=device)
#     encoder_state_dict = {k: v for k, v in checkpoint_ijepa.items() if k in encoder.state_dict()}
#     encoder.load_state_dict(encoder_state_dict, strict=False)

#     classifier = ClassifierHead(encoder, num_classes=4).to(device)  # Ensure 4 classes

#     checkpoint_classifier = torch.load(f"{MODEL_DIR}/best_classifier.pth", map_location=device)
#     if isinstance(checkpoint_classifier, dict):
#         classifier_state_dict = checkpoint_classifier.get('model_state_dict', checkpoint_classifier)
#     else:
#         classifier_state_dict = checkpoint_classifier

#     filtered_classifier_state_dict = {k: v for k, v in classifier_state_dict.items() if k in classifier.state_dict()}
#     classifier.load_state_dict(filtered_classifier_state_dict, strict=False)

#     classifier.eval()
#     return classifier

# classifier = load_models()

# # --- Image preprocessing ---
# IMAGE_SIZE = 224
# transform = transforms.Compose([
#     transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                          std=[0.229, 0.224, 0.225])
# ])

# # --- Grad-CAM class ---
# class GradCAM:
#     def __init__(self, model, target_layer):
#         self.model = model
#         self.target_layer = target_layer
#         self.gradients = None
#         self.activations = None
#         self._register_hooks()

#     def _register_hooks(self):
#         def forward_hook(module, input, output):
#             self.activations = output.detach()

#         def backward_hook(module, grad_input, grad_output):
#             self.gradients = grad_output[0].detach()

#         self.target_layer.register_forward_hook(forward_hook)
#         self.target_layer.register_full_backward_hook(backward_hook)  # use full_backward_hook for compatibility

#     def generate(self, input_tensor, class_idx):
#         self.model.zero_grad()
#         output = self.model(input_tensor)
#         output[0, class_idx].backward()

#         grads = self.gradients[0]       # [C, H, W]
#         acts = self.activations[0]      # [C, H, W]

#         weights = grads.mean(dim=(1, 2), keepdim=True)
#         cam = (weights * acts).sum(dim=0)

#         cam = F.relu(cam)
#         cam -= cam.min()
#         if cam.max() != 0:
#             cam /= cam.max()

#         return cam.cpu().numpy()

# # --- Streamlit UI ---
# st.title("🧠 Brain MRI Classification with Grad-CAM")

# uploaded_file = st.file_uploader("Upload a brain MRI image (jpg/png)", type=['jpg', 'jpeg', 'png'])

# if uploaded_file is not None:
#     image = Image.open(uploaded_file).convert("RGB")
#     st.image(image, caption="🖼️ Uploaded Image", use_column_width=True)

#     input_tensor = transform(image).unsqueeze(0).to(device)

#     with torch.no_grad():
#         outputs = classifier(input_tensor)
#         probs = torch.nn.functional.softmax(outputs, dim=1)
#         pred_idx = probs.argmax(dim=1).item()
#         confidence = probs[0][pred_idx].item()

#     st.markdown(f"### 🧾 Predicted Class: `{CLASS_NAMES[pred_idx]}`")
#     st.markdown(f"**Confidence:** `{confidence:.4f}`")

#     # Grad-CAM
#     target_layer = classifier.encoder.backbone.conv_head
#     gradcam = GradCAM(classifier, target_layer)
#     cam = gradcam.generate(input_tensor, pred_idx)

#     # Overlay heatmap
#     cam_resized = cv2.resize(cam, (image.width, image.height))
#     heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
#     heatmap = np.float32(heatmap) / 255
#     image_np = np.array(image) / 255.0
#     overlay = heatmap + image_np
#     overlay = overlay / np.max(overlay)
#     overlay = np.uint8(255 * overlay)

#     st.image(overlay, caption="🔍 Grad-CAM Heatmap Overlay", use_column_width=True)
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import cv2
import altair as alt
import pandas as pd

# --- Set page configuration as the first Streamlit command ---
st.set_page_config(page_title="🧠 Brain MRI Classifier", layout="wide")

# --- Device configuration ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Translations ---
translations = {
    "en": {
        "title": "🧠 Brain MRI Classifier ",
        "sidebar_title": "🧠 MRI Dementia Classifier",
        "sidebar_desc": "Upload a brain MRI image to classify dementia stage.",
        "sidebar_credit": "Made with ❤️ by Zakaria Arfaoui & Yassine Moumni",
        "upload_prompt": "Upload a brain MRI image (jpg/png)",
        "uploaded_caption": "🖼️ Uploaded Image",
        "blurry_warning": "⚠️ Image may be blurry. Please upload a clearer MRI scan.",
        "prediction_title": "🧾 Predicted Class:",
        "confidence_label": "Confidence:",
        "heatmap_caption": "🔥 Grad-CAM Heatmap Overlay (Aligned)",
        "resized_caption": "🖼️ Resized Input Image",
        "chart_title": "Prediction Confidence"
    },
    "fr": {
        "title": "🧠 Classificateur IRM Cérébrale",
        "sidebar_title": "🧠 Classificateur de Démence par IRM",
        "sidebar_desc": "Téléchargez une image IRM cérébrale pour classifier le stade de démence.",
        "sidebar_credit": "Fabriqué avec ❤️ par Zakaria Arfaoui & Yassine Moumni",
        "upload_prompt": "Téléchargez une image IRM cérébrale (jpg/png)",
        "uploaded_caption": "🖼️ Image Téléchargée",
        "blurry_warning": "⚠️ L'image peut être floue. Veuillez télécharger une IRM plus nette.",
        "prediction_title": "🧾 Classe Prédite :",
        "confidence_label": "Confiance :",
        "heatmap_caption": "🔥 Superposition de la Carte de Chaleur Grad-CAM (Alignée)",
        "resized_caption": "🖼️ Image Redimensionnée",
        "chart_title": "Confiance de la Prédiction"
    },
    "ar": {
        "title": "🧠 تصنيف الرنين المغناطيسي للدماغ",
        "sidebar_title": "🧠 تصنيف الخرف باستخدام الرنين المغناطيسي",
        "sidebar_desc": "قم برفع صورة رنين مغناطيسي للدماغ لتصنيف مرحلة الخرف.",
        "sidebar_credit": "تم إنشاؤه بـ ❤️ بواسطة Zakaria Arfaoui & Yassine Moumni",
        "upload_prompt": "قم برفع صورة رنين مغناطيسي للدماغ (jpg/png)",
        "uploaded_caption": "🖼️ الصورة المرفوعة",
        "blurry_warning": "⚠️ الصورة قد تكون غامضة. يرجى رفع صورة رنين مغناطيسي أوضح.",
        "prediction_title": "🧾 التصنيف المتوقع:",
        "confidence_label": "الثقة:",
        "heatmap_caption": "🔥 طبقة خريطة الحرارة Grad-CAM (محاذاة)",
        "resized_caption": "🖼️ الصورة المعدلة الحجم",
        "chart_title": "ثقة التوقع"
    }
}

# --- Language selection ---
lang = st.sidebar.selectbox("Language / Langue / اللغة", ["en", "fr", "ar"])
t = translations[lang]

# --- Class labels and tips ---
CLASS_NAMES = ["Mild Demented", "Moderate Demented", "Non Demented", "Very Mild Demented"]
CLASS_TIPS = {
    "Mild Demented": "🟡 Tip: Monitor memory and reasoning skills. Consider mild medication and cognitive therapy.",
    "Moderate Demented": "🔴 Tip: Patient may require assistance with daily tasks. Professional care advised.",
    "Non Demented": "🟢 Tip: No signs detected. Keep a healthy lifestyle and schedule regular check-ups.",
    "Very Mild Demented": "🟠 Tip: Early signs present. Monitor closely and begin preventive cognitive support."
}

# --- Vision Transformer Encoder ---
class ViTEncoder(nn.Module):
    def __init__(self, model_name='tf_efficientnetv2_b0', embed_dim=1280):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0)
        self.embed_dim = embed_dim
    def forward(self, x):
        return self.backbone(x)

# --- Classifier Head ---
class ClassifierHead(nn.Module):
    def __init__(self, encoder, num_classes=4):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(encoder.embed_dim, num_classes)
    def forward(self, x):
        features = self.encoder(x)
        logits = self.classifier(features)
        return logits

# --- Model loading ---
MODEL_DIR = "C:/Users/arfao/Desktop/imp_4c_dataset/models"
@st.cache_resource(show_spinner=False)
def load_models():
    encoder = ViTEncoder().to(device)
    checkpoint_ijepa = torch.load(f"{MODEL_DIR}/best_ijepa.pth", map_location=device)
    encoder_state_dict = {k: v for k, v in checkpoint_ijepa.items() if k in encoder.state_dict()}
    encoder.load_state_dict(encoder_state_dict, strict=False)
    classifier = ClassifierHead(encoder, num_classes=4).to(device)
    checkpoint_classifier = torch.load(f"{MODEL_DIR}/best_classifier.pth", map_location=device)
    if isinstance(checkpoint_classifier, dict):
        classifier_state_dict = checkpoint_classifier.get('model_state_dict', checkpoint_classifier)
    else:
        classifier_state_dict = checkpoint_classifier
    filtered_classifier_state_dict = {k: v for k, v in classifier_state_dict.items() if k in classifier.state_dict()}
    classifier.load_state_dict(filtered_classifier_state_dict, strict=False)
    classifier.eval()
    return classifier
classifier = load_models()

# --- Image preprocessing ---
IMAGE_SIZE = 224
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- Grad-CAM class ---
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()
    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)
    def generate(self, input_tensor, class_idx):
        self.model.zero_grad()
        output = self.model(input_tensor)
        output[0, class_idx].backward()
        grads = self.gradients[0]
        acts = self.activations[0]
        weights = grads.mean(dim=(1, 2), keepdim=True)
        cam = (weights * acts).sum(dim=0)
        cam = F.relu(cam)
        cam -= cam.min()
        if cam.max() != 0:
            cam /= cam.max()
        return cam.cpu().numpy()

# --- Streamlit UI ---
st.sidebar.title(t["sidebar_title"])
st.sidebar.markdown(t["sidebar_desc"])
st.sidebar.markdown(t["sidebar_credit"])

uploaded_file = st.file_uploader(t["upload_prompt"], type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    gray = np.array(image.convert('L'))
    fm = cv2.Laplacian(gray, cv2.CV_64F).var()
    if fm < 100:
        st.warning(t["blurry_warning"])

    st.image(image, caption=t["uploaded_caption"], width=300)

    input_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = classifier(input_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        pred_idx = probs.argmax(dim=1).item()
        confidence = probs[0][pred_idx].item()

    st.markdown(f"### {t['prediction_title']} `{CLASS_NAMES[pred_idx]}`")
    st.markdown(f"**{t['confidence_label']}** `{confidence:.4f}`")
    st.info(CLASS_TIPS[CLASS_NAMES[pred_idx]])

    prob_data = pd.DataFrame({
        'Class': CLASS_NAMES,
        'Confidence': probs[0].cpu().numpy()
    })
    chart = alt.Chart(prob_data).mark_bar().encode(
        x=alt.X('Confidence:Q', scale=alt.Scale(domain=[0, 1]), title="Confidence"),
        y=alt.Y('Class:N', sort='-x', title="Class"),
        color=alt.Color('Class:N', legend=None),
        tooltip=['Class', 'Confidence']
    ).properties(width=400, height=200, title=t["chart_title"])
    st.altair_chart(chart, use_container_width=True)

    target_layer = classifier.encoder.backbone.conv_head
    gradcam = GradCAM(classifier, target_layer)
    cam = gradcam.generate(input_tensor, pred_idx)

    resized_image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    image_np = np.array(resized_image).astype(np.float32) / 255.0
    cam_resized = cv2.resize(cam, (IMAGE_SIZE, IMAGE_SIZE))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
    overlay = heatmap * 0.4 + image_np * 0.6
    overlay = np.uint8(255 * overlay)

    col1, col2 = st.columns(2)
    col1.image(resized_image, caption=t["resized_caption"], width=300)
    col2.image(overlay, caption=t["heatmap_caption"], width=300)