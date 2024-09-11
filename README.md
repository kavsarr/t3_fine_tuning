*Bu proje TEKNOFEST 2024 Antalya T3AI Hackathon Yarışması İnce Ayarlama Kategorisi için geliştirilmiştir.*

# T3 LLM Model Instruct
## Codage teami açık kaynakı büyük dil mpdellerine katkı sağlamayı amaçlıyor

````
.
├── training
├── function_calling
├── datasets/
│   ├── turk-egitim-sistemi
│   ├── turk-hukuku
│   ├── surdurulebilirlik
│   └── tarim
├── assets
├── requirements.txt
└── README.md
````

## Takım Adı: Codage
- 👤 Elgun Hasanov
- 👤 Fatulla Bashirov
- 👤 Kavsar Huseynova
- 👤 Mirakram Aghalarov


## Veri Seti Kaynakları
### Türk Eğitim Sistemi
### Türk Hukuku
### Sürdürülebilirlik
### Tarım

## İnce Ayarlama Süreci Başlatma
Batch size: 128
Learning Rate: 4e-5
Epochs: 2


## Sınama Görevi: Fonksiyon Çağırma
1. Get Weather

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

2. Get Time

{
    "type": "function",
    "function": {
        "name": "get_time",
        "description": "Get the current time globally or for a specific location."
    }
}

3. Get Currency

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
}
