from transformers import pipeline
import pandas as pd
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data_path', help='path to train data')
parser.add_argument('-n', '--n_samples', type=int, default=10, help='number of generated samples')
parser.add_argument('-o', '--output_path', type=str, default='./data/', help='path to directory where we will save output file with generated data')
args = parser.parse_args()

def main():

    if args.data_path is None or args.data_path == "":
        print("ERROR you didn't set up data file path")
        return
    
    data_path = Path(args.data_path)
    assert(data_path.exists())

    paraphraser = pipeline("text2text-generation", 
                        model="ramsrigouthamg/t5_paraphraser", 
                        tokenizer="t5-base")

    prompt = "paraphrase: {text} </s>"

    df = pd.read_csv(data_path.absolute())

    accepted_sample_num = 0
    for idx, row in df.iterrows():
        if row['label'] != 1:
            continue 
        
        outputs = paraphraser(prompt.format(text = row['text']), 
                            max_length=128, 
                            num_return_sequences=1, #! number of perefrase generations 
                            do_sample=True,
                            temperature=1.1,
                            top_k=500,
                            top_p=0.95)
        
        output_file_path = Path(args.output_path)
        output_file = Path(f"{output_file_path.absolute()}/augmented_samples.csv")
        file_create = output_file.exists()

        for out in outputs:
            generated = out['generated_text']
            
            # clean result before insert in file
            cleaned_generation = generated.replace(',', '').replace('\n', ' ')
            
            with open(output_file.absolute(), 'a') as fd:
                if not file_create:
                    fd.write("text,label\n")
                fd.write(f"{cleaned_generation},1\n")

        accepted_sample_num += 1
        if accepted_sample_num == args.n_samples:
            break
        
if __name__ == '__main__':
    main()