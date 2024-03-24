from transformers import pipeline

def generate_text(prompt, model_name='EleutherAI/gpt-neo-2.7B', max_length=50):
    gen = pipeline('text-generation', model=model_name)
    generated_text = gen(prompt, max_length=max_length, do_sample=True)
    return generated_text[0]['generated_text']


