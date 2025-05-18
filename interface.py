# import streamlit as st
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import timm
# from PIL import Image
# import torchvision.transforms as transforms
# import numpy as np
# import cv2
# import altair as alt
# import pandas as pd
# import os
# # --- Set page configuration as the first Streamlit command ---
# st.set_page_config(page_title="🧠 Brain MRI Classifier", layout="wide")

# # --- Device configuration ---
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # --- Translations ---
# translations = {
#     "en": {
#         "title": "🧠 Brain MRI Classifier ",
#         "sidebar_title": "🧠 MRI Dementia Classifier",
#         "sidebar_desc": "Upload a brain MRI image to classify dementia stage.",
#         "sidebar_credit": "Made with ❤️ by Zakaria Arfaoui & Yassine Moumni",
#         "upload_prompt": "Upload a brain MRI image (jpg/png)",
#         "uploaded_caption": "🖼️ Uploaded Image",
#         "blurry_warning": "⚠️ Image may be blurry. Please upload a clearer MRI scan.",
#         "prediction_title": "🧾 Predicted Class:",
#         "confidence_label": "Confidence:",
#         "heatmap_caption": "🔥 Grad-CAM Heatmap Overlay (Aligned)",
#         "resized_caption": "🖼️ Resized Input Image",
#         "chart_title": "Prediction Confidence"
#     },
#     "fr": {
#         "title": "🧠 Classificateur IRM Cérébrale",
#         "sidebar_title": "🧠 Classificateur de Démence par IRM",
#         "sidebar_desc": "Téléchargez une image IRM cérébrale pour classifier le stade de démence.",
#         "sidebar_credit": "Fabriqué avec ❤️ par Zakaria Arfaoui & Yassine Moumni",
#         "upload_prompt": "Téléchargez une image IRM cérébrale (jpg/png)",
#         "uploaded_caption": "🖼️ Image Téléchargée",
#         "blurry_warning": "⚠️ L'image peut être floue. Veuillez télécharger une IRM plus nette.",
#         "prediction_title": "🧾 Classe Prédite :",
#         "confidence_label": "Confiance :",
#         "heatmap_caption": "🔥 Superposition de la Carte de Chaleur Grad-CAM (Alignée)",
#         "resized_caption": "🖼️ Image Redimensionnée",
#         "chart_title": "Confiance de la Prédiction"
#     },
#     "ar": {
#         "title": "🧠 تصنيف الرنين المغناطيسي للدماغ",
#         "sidebar_title": "🧠 تصنيف الخرف باستخدام الرنين المغناطيسي",
#         "sidebar_desc": "قم برفع صورة رنين مغناطيسي للدماغ لتصنيف مرحلة الخرف.",
#         "sidebar_credit": "تم إنشاؤه بـ ❤️ بواسطة Zakaria Arfaoui & Yassine Moumni",
#         "upload_prompt": "قم برفع صورة رنين مغناطيسي للدماغ (jpg/png)",
#         "uploaded_caption": "🖼️ الصورة المرفوعة",
#         "blurry_warning": "⚠️ الصورة قد تكون غامضة. يرجى رفع صورة رنين مغناطيسي أوضح.",
#         "prediction_title": "🧾 التصنيف المتوقع:",
#         "confidence_label": "الثقة:",
#         "heatmap_caption": "🔥 طبقة خريطة الحرارة Grad-CAM (محاذاة)",
#         "resized_caption": "🖼️ الصورة المعدلة الحجم",
#         "chart_title": "ثقة التوقع"
#     }
# }

# # --- Language selection ---
# lang = st.sidebar.selectbox("Language / Langue / اللغة", ["en", "fr", "ar"])
# t = translations[lang]

# # --- Class labels and tips ---
# CLASS_NAMES = ["Mild Demented", "Moderate Demented", "Non Demented", "Very Mild Demented"]
# CLASS_TIPS = {
#     "Mild Demented": "🟡 Tip: Monitor memory and reasoning skills. Consider mild medication and cognitive therapy.",
#     "Moderate Demented": "🔴 Tip: Patient may require assistance with daily tasks. Professional care advised.",
#     "Non Demented": "🟢 Tip: No signs detected. Keep a healthy lifestyle and schedule regular check-ups.",
#     "Very Mild Demented": "🟠 Tip: Early signs present. Monitor closely and begin preventive cognitive support."
# }

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
# #MODEL_DIR = "C:/Users/arfao/Desktop/imp_4c_dataset/models"
# MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
# @st.cache_resource(show_spinner=False)
# def load_models():
#     encoder = ViTEncoder().to(device)
#     checkpoint_ijepa = torch.load(f"{MODEL_DIR}/best_ijepa.pth", map_location=device)
#     encoder_state_dict = {k: v for k, v in checkpoint_ijepa.items() if k in encoder.state_dict()}
#     encoder.load_state_dict(encoder_state_dict, strict=False)
#     classifier = ClassifierHead(encoder, num_classes=4).to(device)
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
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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
#         self.target_layer.register_full_backward_hook(backward_hook)
#     def generate(self, input_tensor, class_idx):
#         self.model.zero_grad()
#         output = self.model(input_tensor)
#         output[0, class_idx].backward()
#         grads = self.gradients[0]
#         acts = self.activations[0]
#         weights = grads.mean(dim=(1, 2), keepdim=True)
#         cam = (weights * acts).sum(dim=0)
#         cam = F.relu(cam)
#         cam -= cam.min()
#         if cam.max() != 0:
#             cam /= cam.max()
#         return cam.cpu().numpy()

# # --- Streamlit UI ---
# st.sidebar.title(t["sidebar_title"])
# st.sidebar.markdown(t["sidebar_desc"])
# st.sidebar.markdown(t["sidebar_credit"])

# uploaded_file = st.file_uploader(t["upload_prompt"], type=['jpg', 'jpeg', 'png'])

# if uploaded_file is not None:
#     image = Image.open(uploaded_file).convert("RGB")
#     gray = np.array(image.convert('L'))
#     fm = cv2.Laplacian(gray, cv2.CV_64F).var()
#     if fm < 100:
#         st.warning(t["blurry_warning"])

#     st.image(image, caption=t["uploaded_caption"], width=300)

#     input_tensor = transform(image).unsqueeze(0).to(device)
#     with torch.no_grad():
#         outputs = classifier(input_tensor)
#         probs = torch.nn.functional.softmax(outputs, dim=1)
#         pred_idx = probs.argmax(dim=1).item()
#         confidence = probs[0][pred_idx].item()

#     st.markdown(f"### {t['prediction_title']} `{CLASS_NAMES[pred_idx]}`")
#     st.markdown(f"**{t['confidence_label']}** `{confidence:.4f}`")
#     st.info(CLASS_TIPS[CLASS_NAMES[pred_idx]])

#     prob_data = pd.DataFrame({
#         'Class': CLASS_NAMES,
#         'Confidence': probs[0].cpu().numpy()
#     })
#     chart = alt.Chart(prob_data).mark_bar().encode(
#         x=alt.X('Confidence:Q', scale=alt.Scale(domain=[0, 1]), title="Confidence"),
#         y=alt.Y('Class:N', sort='-x', title="Class"),
#         color=alt.Color('Class:N', legend=None),
#         tooltip=['Class', 'Confidence']
#     ).properties(width=400, height=200, title=t["chart_title"])
#     st.altair_chart(chart, use_container_width=True)

#     target_layer = classifier.encoder.backbone.conv_head
#     gradcam = GradCAM(classifier, target_layer)
#     cam = gradcam.generate(input_tensor, pred_idx)

#     resized_image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
#     image_np = np.array(resized_image).astype(np.float32) / 255.0
#     cam_resized = cv2.resize(cam, (IMAGE_SIZE, IMAGE_SIZE))
#     heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
#     heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
#     overlay = heatmap * 0.4 + image_np * 0.6
#     overlay = np.uint8(255 * overlay)

#     col1, col2 = st.columns(2)
#     col1.image(resized_image, caption=t["resized_caption"], width=300)
#     col2.image(overlay, caption=t["heatmap_caption"], width=300)

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
import os

# Set page configuration for responsiveness
st.set_page_config(
    page_title="🧠 Brain MRI Classifier",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Translations
translations = {
    "en": {
        "title": "🧠 Brain MRI Classifier",
        "sidebar_title": "🧠 MRI Dementia Classifier",
        "sidebar_desc": "Upload brain MRI images to classify dementia stage.",
        "sidebar_credit": "Made with ❤️ by Zakaria Arfaoui & Yassine Moumni",
        "upload_prompt": "Upload brain MRI images (jpg/png)",
        "uploaded_caption": "🖼️ Uploaded Image",
        "blurry_warning": "⚠️ Image may be blurry. Please upload a clearer MRI scan.",
        "prediction_title": "🧾 Predicted Class:",
        "confidence_label": "Confidence:",
        "chart_title": "Prediction Confidence",
        "recommendation_title": "📋 Recommendations"
    },
    "fr": {
        "title": "🧠 Classificateur IRM Cérébrale",
        "sidebar_title": "🧠 Classificateur de Démence par IRM",
        "sidebar_desc": "Téléchargez des images IRM cérébrales pour classifier le stade de démence.",
        "sidebar_credit": "Fabriqué avec ❤️ par Zakaria Arfaoui & Yassine Moumni",
        "upload_prompt": "Téléchargez des images IRM cérébrales (jpg/png)",
        "uploaded_caption": "🖼️ Image Téléchargée",
        "blurry_warning": "⚠️ L'image peut être floue. Veuillez télécharger une IRM plus nette.",
        "prediction_title": "🧾 Classe Prédite :",
        "confidence_label": "Confiance :",
        "chart_title": "Confiance de la Prédiction",
        "recommendation_title": "📋 Recommandations"
    }
}

# Language selection
lang = st.sidebar.selectbox("Language / Langue", ["en", "fr"])
t = translations[lang]

# Class labels, tips, and recommendations
CLASS_NAMES = ["Mild Demented", "Moderate Demented", "Non Demented", "Very Mild Demented"]
CLASS_TIPS = {
    "Mild Demented": "🟡 Tip: Monitor memory and reasoning skills. Consider mild medication and cognitive therapy.",
    "Moderate Demented": "🔴 Tip: Patient may require assistance with daily tasks. Professional care advised.",
    "Non Demented": "🟢 Tip: No signs detected. Keep a healthy lifestyle and schedule regular check-ups.",
    "Very Mild Demented": "🟠 Tip: Early signs present. Monitor closely and begin preventive cognitive support."
}
CLASS_RECOMMENDATIONS = {
    "Mild Demented": [
        "Consult a neurologist for a comprehensive evaluation.",
        "Engage in cognitive training exercises like puzzles or memory games.",
        "Consider medications such as cholinesterase inhibitors as prescribed."
    ],
    "Moderate Demented": [
        "Seek support from a dementia care specialist or professional caregiver.",
        "Implement safety measures at home, such as removing tripping hazards.",
        "Join a support group for patients and caregivers."
    ],
    "Non Demented": [
        "Maintain a brain-healthy diet rich in omega-3s and antioxidants.",
        "Engage in regular physical exercise to promote brain health.",
        "Schedule annual neurological check-ups to monitor brain health."
    ],
    "Very Mild Demented": [
        "Begin cognitive therapy to slow progression of symptoms.",
        "Monitor symptoms closely with regular medical follow-ups.",
        "Explore early intervention programs for dementia prevention."
    ]
}

# Vision Transformer Encoder
class ViTEncoder(nn.Module):
    def __init__(self, model_name='tf_efficientnetv2_b0', embed_dim=1280):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0)
        self.embed_dim = embed_dim
    def forward(self, x):
        return self.backbone(x)

# Classifier Head
class ClassifierHead(nn.Module):
    def __init__(self, encoder, num_classes=4):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(encoder.embed_dim, num_classes)
    def forward(self, x):
        features = self.encoder(x)
        logits = self.classifier(features)
        return logits

# Model loading with robust path handling for Streamlit Cloud
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
@st.cache_resource(show_spinner=False)
def load_models():
    # Define model file paths
    ijepa_path = os.path.join(MODEL_DIR, "best_ijepa.pth")
    classifier_path = os.path.join(MODEL_DIR, "best_classifier.pth")
    
    # Check if model files exist
    if not os.path.exists(ijepa_path) or not os.path.exists(classifier_path):
        st.error(f"Model files not found in {MODEL_DIR}. Please ensure 'best_ijepa.pth' and 'best_classifier.pth' are in the 'models' folder of your GitHub repository.")
        st.stop()
    
    try:
        # Initialize models
        encoder = ViTEncoder().to(device)
        classifier = ClassifierHead(encoder, num_classes=4).to(device)
        
        # Load IJEPA checkpoint (for encoder)
        checkpoint_ijepa = torch.load(ijepa_path, map_location=device)
        # Adjust keys to match encoder.backbone structure
        adjusted_ijepa_state_dict = {}
        for k, v in checkpoint_ijepa.items():
            if not k.startswith("encoder.backbone."):
                new_key = f"backbone.{k}" if not k.startswith("backbone.") else k
                adjusted_ijepa_state_dict[new_key] = v
            else:
                adjusted_ijepa_state_dict[k] = v
        encoder.load_state_dict(adjusted_ijepa_state_dict, strict=False)
        
        # Load classifier checkpoint
        checkpoint_classifier = torch.load(classifier_path, map_location=device)
        # Adjust keys to match classifier structure
        adjusted_classifier_state_dict = {}
        for k, v in checkpoint_classifier.items():
            if k.startswith("conv_stem.") or k.startswith("bn") or k.startswith("blocks."):
                adjusted_classifier_state_dict[f"encoder.backbone.{k}"] = v
            elif k.startswith("classifier."):
                adjusted_classifier_state_dict[k] = v
            else:
                adjusted_classifier_state_dict[k] = v
        classifier.load_state_dict(adjusted_classifier_state_dict, strict=False)
        
        classifier.eval()
        return classifier
    except Exception as e:
        st.error(f"Failed to load models: {str(e)}")
        st.stop()

try:
    classifier = load_models()
except Exception as e:
    st.error(f"Failed to initialize model loading: {str(e)}")
    st.stop()

# Image preprocessing
IMAGE_SIZE = 224
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Responsive CSS for grid and container styling
st.markdown("""
    <style>
    @media (max-width: 600px) {
        .stSidebar {
            width: 100%;
        }
        .stFileUploader {
            width: 100%;
        }
        .stImage {
            width: 100% !important;
            max-width: 100%;
        }
        .stColumn {
            flex: 1;
            min-width: 0;
            padding: 5px;
        }
        .stAltairChart {
            width: 100% !important;
        }
        .square-container, .rectangular-container {
            width: 100% !important;
            margin-bottom: 10px;
            padding: 5px;
        }
        .prediction-text {
            font-size: 14px !important;
        }
        .recommendation-text {
            font-size: 12px !important;
        }
    }
    .square-container {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 5px;
        margin-bottom: 10px;
        background-color: #f9f9f9;
        height: auto;
        min-height: 300px;
        display: flex;
        flex-direction: column;
    }
    .rectangular-container {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 5px;
        margin-bottom: 10px;
        background-color: #f9f9f9;
        height: auto;
        min-height: 250px;
        display: flex;
        flex-direction: column;
    }
    .stColumn {
        padding: 5px;
    }
    .prediction-text {
        font-size: 16px;
    }
    .recommendation-text {
        font-size: 14px;
    }
    /* Reduce Streamlit container margins */
    .element-container {
        margin-top: 0 !important;
        margin-bottom: 0 !important;
    }
    </style>
""", unsafe_allow_html=True)

# Streamlit UI
st.sidebar.title(t["sidebar_title"])
st.sidebar.markdown(t["sidebar_desc"])
st.sidebar.markdown(t["sidebar_credit"])

# File uploader for multiple images
uploaded_files = st.file_uploader(t["upload_prompt"], type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)

if uploaded_files:
    # Reverse uploaded files to show newest at top
    uploaded_files = list(reversed(uploaded_files))
    num_images = len(uploaded_files)
    is_odd = num_images % 2 != 0

    # Process images in pairs for even number or up to last for odd
    for i in range(0, num_images, 2):
        # Create a row with 2 columns
        cols = st.columns(2)

        # First image in the pair (square container)
        with cols[0]:
            if i < num_images:
                with st.container():
                    container_class = "rectangular-container" if is_odd and i == num_images - 1 else "square-container"
                    st.markdown(f'<div class="{container_class}">', unsafe_allow_html=True)
                    
                    # Load and process image
                    image = Image.open(uploaded_files[i]).convert("RGB")
                    gray = np.array(image.convert('L'))
                    fm = cv2.Laplacian(gray, cv2.CV_64F).var()
                    if fm < 100:
                        st.warning(t["blurry_warning"])

                    # Create two columns for image and prediction results
                    col1, col2 = st.columns([1, 1])

                    # Left column: Display uploaded image
                    with col1:
                        st.image(image, caption=t["uploaded_caption"], width=None)

                    # Right column: Display prediction results
                    with col2:
                        # Preprocess image for model
                        input_tensor = transform(image).unsqueeze(0).to(device)
                        with torch.no_grad():
                            outputs = classifier(input_tensor)
                            probs = torch.nn.functional.softmax(outputs, dim=1)
                            pred_idx = probs.argmax(dim=1).item()
                            confidence = probs[0][pred_idx].item()

                        # Display prediction results with responsive font
                        st.markdown(f'<div class="prediction-text">### {t["prediction_title"]} `{CLASS_NAMES[pred_idx]}`</div>', unsafe_allow_html=True)
                        st.markdown(f'<div class="prediction-text">**{t["confidence_label"]}** `{confidence:.4f}`</div>', unsafe_allow_html=True)
                        st.info(CLASS_TIPS[CLASS_NAMES[pred_idx]])

                        # Display recommendations with responsive font
                        st.markdown(f'<div class="prediction-text">### {t["recommendation_title"]}</div>', unsafe_allow_html=True)
                        for rec in CLASS_RECOMMENDATIONS[CLASS_NAMES[pred_idx]]:
                            st.markdown(f'<div class="recommendation-text">- {rec}</div>', unsafe_allow_html=True)

                    # Separate section for confidence chart
                    st.markdown("---")
                    prob_data = pd.DataFrame({
                        'Class': CLASS_NAMES,
                        'Confidence': probs[0].cpu().numpy()
                    })
                    chart = alt.Chart(prob_data).mark_bar().encode(
                        x=alt.X('Confidence:Q', scale=alt.Scale(domain=[0, 1]), title="Confidence"),
                        y=alt.Y('Class:N', sort='-x', title="Class"),
                        color=alt.Color('Class:N', legend=None),
                        tooltip=['Class', 'Confidence']
                    ).properties(height=200, title=t["chart_title"])
                    st.altair_chart(chart, use_container_width=True)

                    st.markdown('</div>', unsafe_allow_html=True)

        # Second image in the pair (square container, if exists)
        with cols[1]:
            if i + 1 < num_images:
                with st.container():
                    container_class = "rectangular-container" if is_odd and i + 1 == num_images - 1 else "square-container"
                    st.markdown(f'<div class="{container_class}">', unsafe_allow_html=True)
                    
                    # Load and process image
                    image = Image.open(uploaded_files[i + 1]).convert("RGB")
                    gray = np.array(image.convert('L'))
                    fm = cv2.Laplacian(gray, cv2.CV_64F).var()
                    if fm < 100:
                        st.warning(t["blurry_warning"])

                    # Create two columns for image and prediction results
                    col1, col2 = st.columns([1, 1])

                    # Left column: Display uploaded image
                    with col1:
                        st.image(image, caption=t["uploaded_caption"], width=None)

                    # Right column: Display prediction results
                    with col2:
                        # Preprocess image for model
                        input_tensor = transform(image).unsqueeze(0).to(device)
                        with torch.no_grad():
                            outputs = classifier(input_tensor)
                            probs = torch.nn.functional.softmax(outputs, dim=1)
                            pred_idx = probs.argmax(dim=1).item()
                            confidence = probs[0][pred_idx].item()

                        # Display prediction results with responsive font
                        st.markdown(f'<div class="prediction-text">### {t["prediction_title"]} `{CLASS_NAMES[pred_idx]}`</div>', unsafe_allow_html=True)
                        st.markdown(f'<div class="prediction-text">**{t["confidence_label"]}** `{confidence:.4f}`</div>', unsafe_allow_html=True)
                        st.info(CLASS_TIPS[CLASS_NAMES[pred_idx]])

                        # Display recommendations with responsive font
                        st.markdown(f'<div class="prediction-text">### {t["recommendation_title"]}</div>', unsafe_allow_html=True)
                        for rec in CLASS_RECOMMENDATIONS[CLASS_NAMES[pred_idx]]:
                            st.markdown(f'<div class="recommendation-text">- {rec}</div>', unsafe_allow_html=True)

                    # Separate section for confidence chart
                    st.markdown("---")
                    prob_data = pd.DataFrame({
                        'Class': CLASS_NAMES,
                        'Confidence': probs[0].cpu().numpy()
                    })
                    chart = alt.Chart(prob_data).mark_bar().encode(
                        x=alt.X('Confidence:Q', scale=alt.Scale(domain=[0, 1]), title="Confidence"),
                        y=alt.Y('Class:N', sort='-x', title="Class"),
                        color=alt.Color('Class:N', legend=None),
                        tooltip=['Class', 'Confidence']
                    ).properties(height=200, title=t["chart_title"])
                    st.altair_chart(chart, use_container_width=True)

                    st.markdown('</div>', unsafe_allow_html=True)