# The Extractive-Abstractive Spectrum: Uncovering Verifiability Trade-offs in LLM Generations

Across all fields of academic study, well-respected experts cite their sources when sharing information. While large language models (LLMs) excel at synthesizing information, they do not provide reliable citation to sources, making it difficult to trace and verify the origins of the information they present. In contrast, search engines make sources readily accessible to users and place the burden of synthesizing information on the user. In our user survey results below, we find that users prefer search engines over LLMs for high-stakes queries, where concerns regarding information provenance outweigh the perceived utility of LLM responses.

<p float="center">
  <img src="/visualize_results/figures/prolific_platform_preference_reasons.png" width="500">
</p>

To rigorously examine the interplay between the verifiability and utility of information sharing tools, we introduce the **extractive-abstractive spectrum** shown below, in which search engines and LLMs are extreme endpoints encapsulating multiple unexplored intermediate operating points. Search engines are **extractive** because they respond to queries with snippets of sources with links (citations) to the original webpages. LLMs are **abstractive** because they address queries with answers that synthesize and logically transform relevant information from training and in-context sources without reliable citation. 

<p float="center">
  <img src="/visualize_results/figures/ea_spectrum_figure_1_robot.png" width="1200">
</p>

We define five operating points that span the extractive-abstractive spectrum and conduct human evaluations on seven systems across four diverse query distributions that reflect real-world QA settings: web search, language simplification, multi-step reasoning, and medical advice. As outputs become more abstractive, our results below demonstrate that perceived utility improves by as much as 200\%, while the proportion of properly cited sentences decreases by as much as 50\% and users take up to 3 times as long to verify cited information. 

In this repository, we share our code to obtain generations from the seven systems, run human evaluation on a [Streamlit](https://streamlit.io/) and [Supabase](https://supabase.com/) webapp, and visualize the results. We also provide our human evaluation data, processing pipeline, and visualization code.

## Code
### Generating cited responses

Update the to-do's in `ea_spectrum.yml` and create the conda environment. 
```
conda env create --name ea_spectrum --file=ea_spectrum.yml
```
This should take under ten minutes. 

Download the Natural Questions (NQ), Wikipedia and Wikidata Multi-Hop (2WikiMH), and Multiple Answer Spans Healthcare (MASH) query datasets to `extractive-abstractive-spectrum/data`.
- NQ: https://ai.google.com/research/NaturalQuestions/download
- 2WikiMH: https://github.com/Alab-NII/2wikimultihop?tab=readme-ov-file
- MASH: https://github.com/mingzhu0527/MASHQA?tab=readme-ov-file

To obtain the snippet, quoted, paraphrased, entailed, and abstractive generations for the NQ (nq), Eta3G (eli5_nq), 2WikiMH (multihop), and MASH (mash) datasets:

```
python citation_systems/generateOperatingPoints.py --start_n 0 --n 20 --project_name example --data nq --best_of_k 10 --retriever google
```
To use gold-standard sources for MASH (`--data mash`) and 2WikiMH (`--data multihop`), replace `--retriever google` with `--gold`.

To obtain the cited gemini generations from files scraped using `save_gemini_html.js` and stored in `example_directory`:

```
python citation_systems/generateGeminiOutputs.py --start_n 0 --n 20 --project_name example --data nq  --html_directory_path example_directory
``` 

To obtain the GPT-4 + Vertex generations:

```
python citation_systems/generatePostHocOutputs.py --start_n 0 --n 20 --project_name example --data nq
```

Each of the scripts above produce a `.json` file of results where each row corresponds to one query with five cited generations. To obtain a file with rows corresponding to individual responses for a query, use the `citation_systems/annotation_processing_for_sl.py` script as described in the next section. None of the example commands above should take more than an hour to run.

### Using the annotation interface
To use the annotation interface, create a new Supabase project and create the tables specified in `annotation_interface/table_schemas.txt`.

Generations from generateOperatingPoint.py, generatePostHocOutputs.py, and generateGeminiOutputs.py must be processed with the following script to obtain example_generations_byQueryOP.csv: 

```
python citation_systems/annotation_processing_for_sl.py --filename example_generations.jsonl --start_n 0 --n 20
```

Instances to annotate from the csv obtained in the previous step must be loaded to Supabase: 

```
python annotation_interface/load_instances_to_annotate.py --filename example_generations_byQueryOP.csv --db instances_to_annotate
```

The same example_generations_byQueryOP.csv needs to be uploaded to Google Drive as a Google Sheets file. Change the Google sheets sharing link to "Anyone with the link" and add it as a gsheets connection in `annotation_interface/.streamlit/secrets.toml`, like so:
```
[connections.gsheets_example]
spreadsheet = "<Google Sheets sharing link>"
```

Credentials and API keys for Supabase and MTurk should also be stored in the `secrets.toml` file. 

Run the app locally by calling `streamlit run annotation_interface/sl_app.py`. The webapp can be deployed through Streamlit.

## Human evaluation data
Processed human evaluation results for the five reference operating point generations (`_ops_`) as well as the GPT-4 + Vertex and Gemini generations (`_baselines_`) are available in `mturk_results/processed_results`. Files ending in `_needs_citation` only contain results for sentences that require citation, as judged by the Vertex API. Files ending in `_all` only contain results for all sentences. The human evaluation results before processing are stored in `mturk_results/unprocessed_results`.

We provide our processing pipeline `visualize_results/clean_mturk_results.ipynb` that obtains the processed result files from the unprocessed results. The processed result files are visualized, as presented in the paper, in `visualize_results/plotting_by_metric.ipynb`. We also provide the processing pipeline as a script: `visualize_results/clean_results.py`.

The user preference survey results are available in `prolific_results` and visualized in `visualize_results/prolific_platform_preference.ipynb`.

## Citation
```
@misc{worledge2024extractiveabstractivespectrumuncoveringverifiability,
      title={The Extractive-Abstractive Spectrum: Uncovering Verifiability Trade-offs in LLM Generations}, 
      author={Theodora Worledge and Tatsunori Hashimoto and Carlos Guestrin},
      year={2024},
      eprint={2411.17375},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2411.17375}, 
}
```

## Contact
[Theodora Worledge](teddiw.github.io)

## License
This project is released under the Apache 2.0 License.
