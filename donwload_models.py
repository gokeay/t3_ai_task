from transformers import T5ForConditionalGeneration, RobertaTokenizer

# İlk model ve tokenizer ayarları
tokenizer_name_1 = "Salesforce/codet5-base-multi-sum"
model_name_1 = "Salesforce/codet5-base-multi-sum"
konum_1 = "weights_all_models/Salesforce_codet5-base-multi-sum"

# Modeli indir ve belirtilen klasöre kaydet
model_1 = T5ForConditionalGeneration.from_pretrained(model_name_1)
model_1.save_pretrained(konum_1)

# Tokenizer'ı indir ve belirtilen klasöre kaydet
tokenizer_1 = RobertaTokenizer.from_pretrained(tokenizer_name_1)
tokenizer_1.save_pretrained(konum_1)

# İkinci model ve tokenizer ayarları
tokenizer_name_2 = "Salesforce/codet5-base"
model_name_2 = "Salesforce/codet5-base-codexglue-sum-javascript"
konum_2 = "weights_all_models/Salesforce_codet5-base-codexglue-sum-javascript"

# Modeli indir ve belirtilen klasöre kaydet
model_2 = T5ForConditionalGeneration.from_pretrained(model_name_2)
model_2.save_pretrained(konum_2)

# Tokenizer'ı indir ve belirtilen klasöre kaydet
tokenizer_2 = RobertaTokenizer.from_pretrained(tokenizer_name_2)
tokenizer_2.save_pretrained(konum_2)
