Model Selection and Implementation
==================================

This section explains why I select specific models and how I implement them.

1. Sentence-Transformer
-----------------------

For the classification I use ``sentence-transformers/distiluse-base-multilingual-cased-v1``.

This model has several advatages compared to the Llama-Model:

* Speed and efficiency: BERT is significantly faster and requires less computational resources than LLaMA models (good for our rexource constrined environment)
* Lower memory requirements: BERT has much lower memory needs, making it more practical for production environments
* Cost-effectiveness: BERT is more affordable to run
* Specialized for classification: BERT's architecture with the [CLS] token makes it naturally suited for classification tasks
* multilinguality for 14 different languages

-> it's lightweight, faster, and specifically designed for tasks that don't require text generation

Implementation details:
-----------------------

The classification pipeline employs an empirically optimized 80:20 ratio of few-shot examples to contextual data, yielding a statistically significant recall improvement 
of ~ Δ=0.15 compared to baseline configurations. A global decision threshold (θ=0.45) is uniformly applied across categories to prevent dataset-specific overfitting while 
preserving model generalizability. Batch processing is constrained to size=4 to mitigate out-of-memory (OOM) exceptions during variable-length sequence processing.

A diagnostic logging pipeline records incentive vectors and top-5 benefit-ranked incentive groups, enabling empirical calibration of the global threshold to optimize 
precision-recall trade-offs. Resource constraints are addressed through post-inference memory deallocation (CUDA cleanup) and CPU offloading protocols.

Further implementation details can be found :func:`here<extraction.classify_incentives_with_few_shot>`.


2. Llama-Model
--------------

For the pure extraction tasks I use ``meta-llama/Llama-3.2-3B-Instruct``.

This model has several advantages compared to other LLMs with simmilar size:

* Specialized retrieval capabilities - Optimized for extracting and summarizing specific information
* Strong reasoning abilities (78.6 on ARC Challenge) - Better understanding of complex incentive structures
* Multilingual support for 8 languages - Process incentives across different language markets
* Lightweight efficiency - Faster processing than larger models with comparable performance (Google, GPT, ...)
* Instruct-Models are trained to follow instructions precisely which limits halucination

Implementation details:
-----------------------

I. General setup for all Llama-Usecases
---------------------------------------

To optimize resource utilization in environments with limited GPU memory (e.g., 8 GB VRAM), several strategies are employed 
during LLM-inference. The parameter ``return_full_text=false`` is consistently set to prevent redundant 
repetition of the input prompt in the output, thereby reducing unnecessary token consumption.

The model is instantiated using mixed precision (float16), which further conserves VRAM without compromising inference quality. 
Post-inference, explicit CUDA memory management routines are invoked to release GPU resources, and the model is dynamically 
loaded and unloaded as needed. These measures collectively enable efficient operation of LLMs in resource-constrained settings.

Prompt engineering follows a standardized structure to maximize model performance and output consistency:

1. A concise description of the overarching task (e.g., identification, classification) and the nature of the input data.
2. A set of clearly enumerated instructions or rules to guide the model’s response.
3. A predefined JSON schema specifying the expected output format.
4. A concluding directive reinforcing strict adherence to the prescribed JSON structure.

This systematic approach ensures both computational efficiency and the reliability of model outputs in constrained computational environments.

II. Experience-Required
-----------------------

The following hyperparameters are systematically calibrated to balance output quality and computational efficiency in constrained-resource environments:

* max_new_tokens=15: Constrains output length to enforce conciseness, reducing hallucination risks and mitigating noise from extraneous token generation.
* temperature=0.1: Optimizes deterministic output by minimizing stochastic sampling (lower bound: 0.01 to avoid gradient collapse).
* repetition_penalty=1.2: Amplified relative to baseline configurations to suppress lexical redundancy in short-sequence outputs.
* do_sample=True: Enables probabilistic sampling for improved solution-space exploration, with trade-offs in deterministic reproducibility requiring rigorous output validation.

The models response in json-format is handeled in :func:`this<extraction.parse_json_response>` function.

JSON-formatted responses are programmatically validated via a :func:`<parsing-function>extraction.parse_json_response`, ensuring syntactic and semantic adherence to predefined schemas.

III. Incentives Extraction
--------------------------

The inference pipeline employs the following empirically derived hyperparameters to maximize throughput while maintaining output integrity under VRAM constraints:

* max_new_tokens=350: Empirically determined upper bound for token generation given VRAM capacity. Truncation artifacts in JSON outputs are mitigated via post-processing with a dedicated :func:json-repair<extraction.parse_json_incentives> utility.
* repetition_penalty=1.1: Supplement to structural prompt rules (Rule 4) to address edge cases of redundant incentive enumeration.
* do_sample=False: Enables deterministic greedy decoding, ensuring reproducible performance metrics during validation phases.

JSON-formatted incentive lists undergo schema validation and lexical sanitization through :func:`<this>extraction.parse_json_incentives` function. 

The model generally is showing good performance values for the incentive extraction. It can be improved by using a larger Llama-model or
finetuning it on some data. See the :ref:`performance section<Performance Metrics>` to see the individual accuiracy and recall per category.

A training of the Llama Model not possible to conduct for me due to my restricted :ref:`GPU<Used Hardware/Software>`.
Training a Llama-3.2-3B-Instruct model with 8 GB VRAM is infeasible due to hardware limitations: 
even quantized 3B models require >12 GB VRAM for training gradients and optimizer states, while inference alone consumes ~6 GB.

IV. Branche
-----------

The following hyperparameters are optimized for categorical classification tasks under VRAM constraints:

* max_new_tokens=20: Enforces output brevity to reduce hallucination risks and token-space noise.
* temperature=0.1: Balances deterministic output with minimal stochasticity; lower thresholds (e.g., 0.01) induce gradient collapse (see :ref:Experience-Required<II. Experience-Required>).
* repetition_penalty=1.1: Suppresses lexical redundancy in compact categorical outputs.
* do_sample=True: Probabilistic sampling increases recall for rare categories.
* top_p=0.95: Nucleus sampling complements stochastic exploration while maintaining output coherence.

Empirical results in the :ref:Performance-Section<Performance Metrics> demonstrate:

1. High Precision: Robust alignment with ground-truth categories across diverse job titles.
2. Edge Case Limitations: Semantic ambiguity in underspecified queries (e.g., atypical job titles) occasionally induces misclassification.




