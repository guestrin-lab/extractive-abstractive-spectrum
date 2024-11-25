# extractive-abstractive-spectrum

### Generating cited responses
To obtain the snippet, quoted, paraphrased, entailed, and abstractive generations for the NQ (nq), Eta3G (eli5_nq), 2WikiMH (multihop), and MASH (mash) datasets:\
`python citation_systems/generateOperatingPoints.py --start_n 0 --n 20 --project_name example --data nq --best_of_k 10`\
To use gold-standard sources for MASH (`--data mash`) and 2WikiMH (`--data multihop`), also add `--gold`.

To obtain the scraped gemini generations (using `save_gemini_html.js` and stored in `example_directory`):\
`python citation_systems/generateGeminiOutputs.py --start_n 0 --n 20 --project_name example --data nq  --html_directory_path example_directory` 

To obtain the GPT-4 + Vertex generations:\
`python citation_systems/generatePostHocOutputs.py --start_n 0 --n 20 --project_name example --data nq` 

### Using the annotation interface
First, generations from generateOperatingPoint.py, generatePostHocOutputs.py, and generateGeminiOutputs.py must be processed with the following script: \
`python annotation_interface/annotation_processing_for_sl.py --filename example_generations.jsonl --start_n 0 --n 20`

Next, instances to annotate from the csv obtained in the previous step must be loaded to Supabase: \
`python annotation_interface/load_instances_to_annotate.py --filename example_generations.csv --db example_instances_to_annotate`

### Data
Coming soon!
