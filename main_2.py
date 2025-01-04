import time
from transformers import T5ForConditionalGeneration, RobertaTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, T5Tokenizer
import torch

tokenizer = RobertaTokenizer.from_pretrained("Salesforce_codet5-base")
model = T5ForConditionalGeneration.from_pretrained("Salesforce_codet5-base-codexglue-sum-javascript")

## Özetlenecek kod girilir
code = """

function calculateSquares(numbers) {
    const result = {};

    numbers.forEach(number => {
        if (typeof number === 'number') {
            result[number] = number * number;
        } else {
            console.warn(`${number} is not a number`);
        }
    });

    return result;
}

console.log(squares);
"""

# code = """

# // program to find the largest among three numbers

# // take input from the user
# const num1 = parseFloat(prompt("Enter first number: "));
# const num2 = parseFloat(prompt("Enter second number: "));
# const num3 = parseFloat(prompt("Enter third number: "));
# let largest;

# // check the condition
# if(num1 >= num2 && num1 >= num3) {
#     largest = num1;
# }
# else if (num2 >= num1 && num2 >= num3) {
#     largest = num2;
# }
# else {
#     largest = num3;
# }

# // display the result
# console.log("The largest number is " + largest);

# """


# code = """

# const num1 = parseFloat(prompt("Enter first number: "));
# const num2 = parseFloat(prompt("Enter second number: "));
# const num3 = parseFloat(prompt("Enter third number: "));
# let largest;

# if(num1 >= num2 && num1 >= num3) {
#     largest = num1;
# }
# else if (num2 >= num1 && num2 >= num3) {
#     largest = num2;
# }
# else {
#     largest = num3;
# }

# console.log("The largest number is " + largest);

# """


print("\n\n")

# GPU ile çalıştırma
if torch.cuda.is_available():
    model.cuda()  # Modeli GPU'ya taşı
    start_time = time.time()  # Zaman ölçümü başlar
    input_ids = tokenizer(code, return_tensors='pt').input_ids.cuda()  # GPU için input'u taşı
    outputs_gpu = model.generate(input_ids, max_new_tokens=300, 
                                 repetition_penalty=1.2,  # Tekrar cezalandırması
                                 no_repeat_ngram_size=3 # Tekrar eden n-gram'ların boyutu
                                )
    gpu_time = time.time() - start_time  # Zaman ölçümü biter
    print("\n\nGPU Result:", tokenizer.decode(outputs_gpu[0], skip_special_tokens=True))
    print(f"GPU execution time: {gpu_time:.2f} seconds")
else:
    print("CUDA is not available. GPU test skipped.")
