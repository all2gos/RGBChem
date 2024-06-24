'logimport torch, os
from PIL import Image
from scripts.params import LOG_FILE, TEST_DIR_NAME, PATH

def load_model(model_path):
    model = torch.load(model_path)
    model.eval()  # Ustawienie modelu w tryb ewaluacji
    return model

# Funkcja wykonująca predykcję
def predict(model, image_path, transform):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Dodanie wymiaru batcha
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    return predicted.item()

# Ścieżka do pliku modelu i obrazu
model_path = 'model.pth'
image_path = 'example_image.jpg'

# Transformacje obrazu
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Dopasowanie rozmiaru obrazu
    transforms.ToTensor(),  # Konwersja obrazu do tensora
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalizacja
])


model = load_model(LOG_FILE.replace('log','pth'))

prediction = predict(model, image_path, transform)
print(f'Predykcja: {prediction}')
