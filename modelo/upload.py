from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Ruta local del modelo y tokenizador
ruta_modelo = "./modelo_clasificacion_sentimientos"
repo_id = "profesorJorgeBaron/prueba"

# Cargar el modelo y el tokenizer desde la ruta local
modelo = AutoModelForSequenceClassification.from_pretrained(ruta_modelo)
tokenizer = AutoTokenizer.from_pretrained(ruta_modelo)

# Subir al repositorio de Hugging Face
modelo.push_to_hub(repo_id)
tokenizer.push_to_hub(repo_id)

print("✅ Modelo y tokenizador subidos correctamente a Hugging Face Hub.")