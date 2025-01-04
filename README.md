# T3 AI Projesi - 2024

## Projenin Amacı
Bu projenin amacı, internetten bağımsız olarak lokal ortamda **JavaScript** kodlarını analiz edebilen ve metin girdisiyle kod üretebilen **CodeT5** tabanlı bir **LLM** modeli geliştirmektir. Proje kapsamında, iki farklı modelin kurulumu, çalıştırılması ve değerlendirilmesi gerçekleştirilmiştir. Modeller, temel görevleri yerine getirebilmekte olup performans açısından birbirleriyle karşılaştırılmıştır. Daha iyi sonuçlar için **fine-tuning** yapılması önerilmektedir.

## Projede Kullanılan Modeller
### Model 1: `Salesforce/codet5-base-multi-sum`
- **Model:** `T5ForConditionalGeneration`
- **Tokenizer:** `Salesforce/codet5-base-multi-sum` | `RobertaTokenizer`
- **Görev:** Girilen kodun analizi (**code-to-text**)
- **Link:** [Hugging Face - CodeT5 Multi-Sum](https://huggingface.co/Salesforce/codet5-base-multi-sum)

### Model 2: `Salesforce/codet5-base-codexglue-sum-javascript`
- **Model:** `T5ForConditionalGeneration`
- **Tokenizer:** `Salesforce/codet5-base` | `RobertaTokenizer`
- **Görev:** Girilen metne göre kod üretilmesi (**text-to-code**)
- **Link:** [Hugging Face - CodeT5 JavaScript](https://huggingface.co/Salesforce/codet5-base-codexglue-sum-javascript)

## Projenin Bilgisayara Kurulumu
**Github Linki:** [T3 AI Projesi - Github](https://github.com/gokeay/hvlsn_2024)

### Adımlar:
1. Proje klonlanır ve proje dizinine girilir:
    ```bash
    git clone https://github.com/gokeay/hvlsn_2024
    cd hvlsn_2024
    ```

2. Sanal ortam oluşturulur ve etkinleştirilir (isteğe bağlı):
    ```bash
    python -m venv myenv
    .\myenv\Scripts\activate
    ```

3. Gerekli bağımlılıklar indirilir:
    ```bash
    pip install -r requirements.txt
    ```

4. Modeller, sırasıyla aşağıdaki dosyalardan çalıştırılabilir:
   - **Model 1:** `main.py`
   - **Model 2:** `main_2.py`

## Notlar
1. Modellerin performansını karşılaştırmak için örnek görevler üzerinde testler yapılmıştır.
2. Modeli JavaScript özelinde eğitmek için 60k JavaScript ve özet içeriği: [Hugging Face Dataset](https://huggingface.co/datasets/google/code_x_glue_ct_code_to_text/viewer/javascript?row=7)
