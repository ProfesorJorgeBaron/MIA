from transformers import pipeline
import gradio as gr


clasificador = pipeline("sentiment-analysis", model="pysentimiento/robertuito-sentiment-analysis")

def ver_sentimientos(frase):
    respuesta = clasificador(frase)
    print(respuesta)
    if(respuesta[0]['label'] == "POS"):
        resultado= "POSITIVO"   
    else:
        resultado = "NEGATIVO"
    return resultado

demo = gr.Interface(fn=ver_sentimientos, inputs="text", outputs="text")
demo.launch()