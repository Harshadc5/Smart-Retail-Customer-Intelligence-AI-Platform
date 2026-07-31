# The "Notebook-as-Generator" Architecture

Instead of utilizing a standard static repository, the Smart Retail AI platform utilizes a dynamic Jupyter Notebook (`Smart_Retail_Final_Project.ipynb`) as a source-of-truth builder to generate the backend.

## The problem

In heavily academic or prototyped AI projects, data scientists often build messy, monolithic Jupyter notebooks. When it is time to deploy, engineers must manually extract the models and rewrite the API layer. This causes a massive disconnect between the experimental data science phase and the production engineering phase.

## The approach

By utilizing python's `os` and file I/O operations directly inside the notebook, the notebook itself *becomes* the compiler. When executed, it programmatically writes `app/main.py`, `app/routers/vision.py`, and serializes the `.pkl` and `.h5` files into a strict Domain-Driven Design (DDD) directory structure.

## Trade-offs

- **Pros**: The entire project (data preparation, model training, and API generation) is reproducible with a single "Run All" command.
- **Cons**: Standard Git diffs are difficult to read for `.ipynb` files, making collaborative branch-merging slightly more tedious than editing static `.py` files.
