# extractive-abstractive-spectrum

### Generating cited responses
To obtain the snippet, quoted, paraphrased, entailed, and abstractive generations for the NQ (nq), Eta3G (eli3), 2WikiMH (mh), and MASH (mash) datasets:\
`python generateOperatingPoints.py --start_n 0 --n 20 --project_name example --data nq --best_of_k 10 # For MASH (--data mash) and 2WikiMH (--data mh), also use: --gold` 

To obtain the scraped gemini generations (using `save_gemini_html.js` and stored in `example_directory`):\
`python generateGeminiOutputs.py --start_n 0 --n 20 --project_name example --data nq  --html_directory_path example_directory` 

To obtain the GPT-4 + Vertex generations:\
`python generatePostHocOutputs.py --start_n 0 --n 20 --project_name example --data nq` 

### Using the annotation interface
First, generations from generateOperatingPoint.py, generatePostHocOutputs.py, and generateGeminiOutputs.py must be processed with the following script: 

`python annotation_processing_for_sl.py --filename example_generations.jsonl --start_n 0 --n 20`
