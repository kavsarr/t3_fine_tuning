from transformers import pipeline

prompt = """Aşağıdaki fonksiyonlar verildiğinde, lütfen verilen isteme en iyi şekilde cevap verecek bir fonksiyon çağrısı için uygun argümanlarla bir JSON döndürün.

{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather conditions for a specific location.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g., Shaoxing, China"
                }
            },
            "required": ["location"]
        }
    }
}

{
    "type": "function",
    "function": {
        "name": "get_time",
        "description": "Get the current time globally or for a specific location."
    }
}

{
    "type": "function",
    "function": {
        "name": "get_currency",
        "description": "Convert a value from one currency to another.",
        "parameters": {
            "type": "object",
            "properties": {
                "from": {
                    "type": "string",
                    "description": "The currency code to convert from, e.g., TRY"
                },
                "to": {
                    "type": "string",
                    "description": "The currency code to convert to, e.g., EUR"
                },
                "value": {
                    "type": "number",
                    "description": "The amount to convert, e.g., 57000"
                }
            },
            "required": ["from", "to", "value"]
        }
    }
}"""

question = "Antalyada hava nasıl"
    
messages = [
    {"role": "user", "content": prompt+f"\nSoru: {question}"}
    ]

pipe = pipeline("text-generation", model="cp/tools/final", device='cuda')

print(pipe(messages))