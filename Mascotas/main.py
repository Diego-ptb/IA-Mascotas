from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import io
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions

app = FastAPI(title="API de Reconocimiento Visual de Mascotas")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

model_vector = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')
model_classifier = MobileNetV2(weights='imagenet', include_top=True)

# Clases de ImageNet que corresponden a mascotas (nombres en minúsculas con guiones bajos)
PET_CLASSES = {
    # Perros — todas las razas reconocidas por ImageNet/MobileNetV2
    'chihuahua', 'japanese_spaniel', 'maltese_dog', 'pekinese', 'shih-tzu',
    'blenheim_spaniel', 'papillon', 'toy_terrier', 'rhodesian_ridgeback',
    'afghan_hound', 'basset', 'beagle', 'bloodhound', 'bluetick',
    'black-and-tan_coonhound', 'walker_hound', 'english_foxhound', 'redbone',
    'borzoi', 'irish_wolfhound', 'italian_greyhound', 'whippet',
    'ibizan_hound', 'norwegian_elkhound', 'otterhound', 'saluki',
    'scottish_deerhound', 'weimaraner', 'staffordshire_bullterrier',
    'american_staffordshire_terrier', 'bedlington_terrier', 'border_terrier',
    'kerry_blue_terrier', 'irish_terrier', 'norfolk_terrier', 'norwich_terrier',
    'yorkshire_terrier', 'wire-haired_fox_terrier', 'lakeland_terrier',
    'sealyham_terrier', 'airedale', 'cairn', 'australian_terrier',
    'dandie_dinmont', 'boston_bull', 'miniature_schnauzer', 'scotch_terrier',
    'tibetan_terrier', 'silky_terrier', 'soft-coated_wheaten_terrier',
    'west_highland_white_terrier', 'lhasa', 'flat-coated_retriever',
    'curly-coated_retriever', 'golden_retriever', 'labrador_retriever',
    'chesapeake_bay_retriever', 'german_short-haired_pointer', 'vizsla',
    'english_setter', 'irish_setter', 'gordon_setter', 'brittany_spaniel',
    'clumber', 'english_springer', 'welsh_springer_spaniel', 'cocker_spaniel',
    'sussex_spaniel', 'irish_water_spaniel', 'kuvasz', 'schipperke',
    'groenendael', 'malinois', 'briard', 'kelpie', 'komondor',
    'old_english_sheepdog', 'shetland_sheepdog', 'collie', 'border_collie',
    'bouvier_des_flandres', 'rottweiler', 'german_shepherd', 'doberman',
    'miniature_pinscher', 'greater_swiss_mountain_dog', 'bernese_mountain_dog',
    'appenzeller', 'entlebucher', 'boxer', 'bull_mastiff', 'tibetan_mastiff',
    'french_bulldog', 'great_dane', 'saint_bernard', 'eskimo_dog', 'malamute',
    'siberian_husky', 'affenpinscher', 'basenji', 'pug', 'leonberg',
    'newfoundland', 'great_pyrenees', 'samoyed', 'pomeranian', 'chow',
    'keeshond', 'brabancon_griffon', 'pembroke', 'cardigan', 'toy_poodle',
    'miniature_poodle', 'standard_poodle', 'mexican_hairless', 'dingo',
    'dhole', 'african_hunting_dog', 'akita',
    # Gatos
    'tabby', 'tiger_cat', 'persian_cat', 'siamese_cat', 'egyptian_cat',
    # Otras mascotas comunes
    'rabbit', 'hamster', 'guinea_pig', 'parrot', 'macaw', 'cockatoo',
    'parakeet', 'lorikeet', 'goldfish', 'turtle',
}

DICCIONARIO_RAZAS = {
    # --- Perros ---
    'golden_retriever': 'Golden Retriever',
    'labrador_retriever': 'Labrador Retriever',
    'chesapeake_bay_retriever': 'Chesapeake Bay Retriever',
    'flat-coated_retriever': 'Retriever de Pelo Liso',
    'curly-coated_retriever': 'Retriever de Pelo Rizado',
    'german_shepherd': 'Pastor Alemán',
    'pug': 'Pug (Carlino)',
    'beagle': 'Beagle',
    'french_bulldog': 'Bulldog Francés',
    'boston_bull': 'Boston Terrier',
    'rottweiler': 'Rottweiler',
    'doberman': 'Dobermann',
    'boxer': 'Boxer',
    'siberian_husky': 'Husky Siberiano',
    'malamute': 'Malamute de Alaska',
    'eskimo_dog': 'Perro Esquimal',
    'samoyed': 'Samoyedo',
    'pomeranian': 'Pomerania',
    'chow': 'Chow Chow',
    'akita': 'Akita Inu',
    'shih-tzu': 'Shih Tzu',
    'chihuahua': 'Chihuahua',
    'maltese_dog': 'Maltés',
    'pekinese': 'Pequinés',
    'lhasa': 'Lhasa Apso',
    'papillon': 'Papillón',
    'japanese_spaniel': 'Spaniel Japonés (Chin)',
    'blenheim_spaniel': 'Spaniel Blenheim',
    'cocker_spaniel': 'Cocker Spaniel',
    'english_springer': 'Springer Spaniel Inglés',
    'welsh_springer_spaniel': 'Springer Spaniel Galés',
    'irish_water_spaniel': 'Spaniel de Agua Irlandés',
    'sussex_spaniel': 'Sussex Spaniel',
    'clumber': 'Clumber Spaniel',
    'brittany_spaniel': 'Epagneul Bretón',
    'border_collie': 'Border Collie',
    'collie': 'Collie',
    'shetland_sheepdog': 'Shetland Sheepdog',
    'old_english_sheepdog': 'Bobtail (Ovejero Inglés)',
    'pembroke': 'Corgi Pembroke',
    'cardigan': 'Corgi Cardigan',
    'kelpie': 'Kelpie Australiano',
    'groenendael': 'Groenendael (Pastor Belga Negro)',
    'malinois': 'Malinois (Pastor Belga)',
    'briard': 'Briard (Pastor de Brie)',
    'bouvier_des_flandres': 'Bouvier de Flandes',
    'great_dane': 'Gran Danés',
    'saint_bernard': 'San Bernardo',
    'newfoundland': 'Terranova',
    'leonberg': 'Leonberger',
    'great_pyrenees': 'Gran Pirineo',
    'bernese_mountain_dog': 'Boyero de Berna',
    'greater_swiss_mountain_dog': 'Gran Boyero Suizo',
    'appenzeller': 'Appenzeller',
    'entlebucher': 'Entlebucher',
    'kuvasz': 'Kuvasz',
    'komondor': 'Komondor',
    'tibetan_mastiff': 'Mastín Tibetano',
    'bull_mastiff': 'Bullmastiff',
    'weimaraner': 'Weimaraner',
    'vizsla': 'Vizsla (Braco Húngaro)',
    'german_short-haired_pointer': 'Braco Alemán de Pelo Corto',
    'english_setter': 'Setter Inglés',
    'irish_setter': 'Setter Irlandés',
    'gordon_setter': 'Setter Gordon',
    'afghan_hound': 'Galgo Afgano',
    'borzoi': 'Borzói (Galgo Ruso)',
    'italian_greyhound': 'Galgo Italiano',
    'whippet': 'Whippet',
    'saluki': 'Saluki',
    'scottish_deerhound': 'Lebrel Escocés',
    'irish_wolfhound': 'Lobero Irlandés',
    'ibizan_hound': 'Podenco Ibicenco',
    'basset': 'Basset Hound',
    'bloodhound': 'Sabueso',
    'bluetick': 'Bluetick Coonhound',
    'redbone': 'Redbone Coonhound',
    'black-and-tan_coonhound': 'Coonhound Negro y Fuego',
    'walker_hound': 'Treeing Walker Coonhound',
    'english_foxhound': 'Foxhound Inglés',
    'rhodesian_ridgeback': 'Rhodesian Ridgeback',
    'basenji': 'Basenji',
    'norwegian_elkhound': 'Cazador de Alces Noruego',
    'otterhound': 'Otterhound',
    'toy_poodle': 'Caniche Toy',
    'miniature_poodle': 'Caniche Miniatura',
    'standard_poodle': 'Caniche Estándar',
    'miniature_schnauzer': 'Schnauzer Miniatura',
    'standard_schnauzer': 'Schnauzer Estándar (no existe en ImageNet, alias)',
    'giant_schnauzer': 'Schnauzer Gigante (no existe en ImageNet, alias)',
    'airedale': 'Airedale Terrier',
    'scotch_terrier': 'Terrier Escocés',
    'west_highland_white_terrier': 'West Highland White Terrier',
    'cairn': 'Cairn Terrier',
    'norfolk_terrier': 'Norfolk Terrier',
    'norwich_terrier': 'Norwich Terrier',
    'yorkshire_terrier': 'Yorkshire Terrier',
    'lakeland_terrier': 'Lakeland Terrier',
    'wire-haired_fox_terrier': 'Fox Terrier de Pelo Duro',
    'staffordshire_bullterrier': 'Staffordshire Bull Terrier',
    'american_staffordshire_terrier': 'American Staffordshire Terrier',
    'bedlington_terrier': 'Bedlington Terrier',
    'border_terrier': 'Border Terrier',
    'kerry_blue_terrier': 'Kerry Blue Terrier',
    'irish_terrier': 'Terrier Irlandés',
    'australian_terrier': 'Terrier Australiano',
    'silky_terrier': 'Terrier Sedoso',
    'soft-coated_wheaten_terrier': 'Terrier de Trigo',
    'sealyham_terrier': 'Sealyham Terrier',
    'dandie_dinmont': 'Dandie Dinmont Terrier',
    'tibetan_terrier': 'Terrier Tibetano',
    'toy_terrier': 'Toy Terrier',
    'miniature_pinscher': 'Pinscher Miniatura',
    'affenpinscher': 'Affenpinscher',
    'brabancon_griffon': 'Grifón de Brabante',
    'keeshond': 'Keeshond',
    'schipperke': 'Schipperke',
    'pomeranian': 'Pomerania',
    'mexican_hairless': 'Xoloitzcuintli (Perro sin pelo mexicano)',
    'dingo': 'Dingo',
    'dhole': 'Perro Salvaje Asiático',
    'african_hunting_dog': 'Perro Salvaje Africano',
    # --- Gatos ---
    'tabby': 'Gato Atigrado',
    'tiger_cat': 'Gato Tigre',
    'persian_cat': 'Gato Persa',
    'siamese_cat': 'Gato Siamés',
    'egyptian_cat': 'Gato Egipcio (Mau)',
    # --- Otras mascotas ---
    'rabbit': 'Conejo',
    'hamster': 'Hámster',
    'guinea_pig': 'Cobayo (Cuy)',
    'parrot': 'Loro',
    'macaw': 'Guacamayo',
    'cockatoo': 'Cacatúa',
    'parakeet': 'Periquito',
    'lorikeet': 'Lori',
    'goldfish': 'Pez Dorado',
    'turtle': 'Tortuga',
}


def es_mascota(class_name: str) -> bool:
    """Devuelve True si el nombre de clase de ImageNet corresponde a una mascota."""
    return class_name.lower() in PET_CLASSES


def traducir(nombre_en):
    return DICCIONARIO_RAZAS.get(nombre_en, nombre_en.replace('_', ' ').title())


def prepare_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB').resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)


def calcular_similitud(v1, v2):
    v1, v2 = np.array(v1), np.array(v2)
    dot_product = np.dot(v1, v2)
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    if norm == 0:
        return 0.0
    return float((dot_product / norm) * 100)


@app.post("/analyze")
async def analyze_image(
    file: UploadFile = File(...),
    existing_vectors: str = Form("[]")
):
    try:
        bytes_data = await file.read()
        input_tensor = prepare_image(bytes_data)

        preds = model_classifier.predict(input_tensor)
        # Obtenemos el Top 5 para buscar la mejor predicción de mascota
        top_preds = decode_predictions(preds, top=5)[0]

        # Buscar la primera predicción que sea una mascota reconocida
        pet_pred = None
        for pred in top_preds:
            if es_mascota(pred[1]):
                pet_pred = pred
                break

        if pet_pred is None:
            return {
                "error": "No se detectó una mascota en la imagen. "
                         "Por favor, sube una foto de un perro, gato u otra mascota."
            }

        raza_en = pet_pred[1]
        confianza_raza = round(float(pet_pred[2]) * 100, 2)

        vector_nuevo = model_vector.predict(input_tensor)[0].tolist()

        try:
            registry = json.loads(existing_vectors)
        except json.JSONDecodeError:
            return {"error": "El formato de existing_vectors no es un JSON válido."}

        matches = []
        for item in registry:
            similitud = calcular_similitud(vector_nuevo, item['vector'])
            matches.append({
                "id": item['id'],
                "similarity": round(similitud, 2)
            })
        matches.sort(key=lambda x: x['similarity'], reverse=True)

        return {
            "ai_prediction": {
                "breed_es": traducir(raza_en),
                "breed_en": raza_en,
                "confidence": confianza_raza
            },
            "new_vector": vector_nuevo,
            "database_matches": matches
        }
    except Exception as e:
        return {"error": f"Ocurrió un error al procesar la imagen: {str(e)}"}
