
# Masterthesis: Informationsextraktion aus Wissensgraphen: Eine Untersuchung moderner Retrieval-Methoden

This project contains all code related to my master thesis. If you want to run the experiments you have to download the required data first. Afterwards a set up of Elastsearch is required. To ensure reproducability it is recommended to use the provided Docker configuration for the Jupyter Server. The Jupyter configuration does not support CUDA at the moment but I strongly recommend it for the ingest to the vector index if you have a GPU.

## Table of Contents
- [Abstract](#abstract)
- [Data](#Data)
- [Data Ingest](#data-ingest)
- [Experiments](#experiments)
- [Analysis](#analysis)

## Abstract

#### Purpose: 
This thesis investigates the benefits of community search (CS) algorithms for refining the results of information retrieval (IR) methods in knowledge graphs. The aim is to evaluate the performance of IR systems after the integration of CS methods, to better identify relevant documents and to increase the quality of search results.
#### Value: 
The combination of IR methods and CS algorithms offers the scientific community new approaches to exploit the hidden potential of knowledge graphs. The results provide insights into the use of knowledge graphs in IR and offer practical implications for the development of future-proof IR systems.
#### Methods: 
Classical (Boolean, probabilistic) and modern (vector-based, hybrid) IR methods were investigated in combination with different CS algorithms in two knowledge graphs of different density and ontology. An evaluation was carried out using established metrics such as NDCG, MAP, precision and recall. Furthermore, the factors influencing the results of the IR systems were analyzed.
#### Most important results: 
Fundamentally, the results confirm the complementary use of semantic and term-based IR approaches as the best performing of those examined. The application of CS algorithms in an IRS shows that by extracting subgraphs in the knowledge graphs, additional relevant documents can be identified that were not captured in the initial IR. It also shows that vector-based IR benefits more from the combination with CS methods in a topic-based knowledge graph than term-based IR methods. Finally, the factors influencing the results of CS were analyzed, which consist of the initial node selection as well as the ontology and the density of the graph. 
#### Conclusion: 
The study shows that the use of a CS has the potential to increase the performance of IRS. In order to exploit the full potential and suppress negative edge effects, further development of the implementations used is required. Furthermore, an investigation of CS in knowledge graphs with directed and weighted edges is required.
#### Keywords: 
information retrieval, knowledge graph, community search, hybrid models, information retrieval system

## Data
### Download
Due to the size of the raw data it is not contained in the repository. In the folder 01_data_preparation there is a notebook [Data_Download](01_data_preparation/Data_Download.ipynb) that contains all code that is needed to download all required data. Alternatively the preprocessed data (Output of [Data_Preprocessing](01_data_preparation/Data_Preprocessing.ipynb)) is available on Kaggle. Data will be stored in 03_data/raw_data.

### Preprocessing
All preprocessing steps are done in [Data_Preprocessing](01_data_preparation/Data_Preprocessing.ipynb). The output was uploaded to [Kaggle]().
Data will be stored in 03_data/preprocessed_data.

## Data Ingest
### Elastic Setup
This chapter is for the data ingest to Elasticsearch (ES). ES is used for BM25, Boolean and Vector Search. ES is set up in docker. All necessary files are located in elastic_search folder. You have to create an API key on initial startup for the Retrieval in the experiment section. Save it in an environment variable called ES_API_KEY.

### Ingestion 
The ingestion files are splitted for [Text Index](02_ingest/Text-Index.ipynb) and [Vector Index](02_ingest/Vector-Index.ipynb). One index will be created for a text index (used by boolean and probabilistic IR) and another for a vector index (used by vector IR). 

## Experiments
The experiments are executed in a Jupyter Environment.
### RQ1
The experiment for RQ1 is located in [RQ1](06_experiment/RQ1/RQ1.ipynb). A running ES cluster with completed data ingest is needed for the execution.
### RQ2
The experiment for RQ2 is located in [RQ2](06_experiment/RQ2/RQ2.ipynb). ES is not needed for this step. The experiment is only set up for one configuration. The configuration can be adjusted to match all other cases that were reported in the thesis.
### Results
Results are saved in 05_results as json. There are different subfolders for reranked and community search
File names are built in the following structure:
- KNOWLEDGE-GRAPH_results_rq2_cs_CS-METHOD_NUMBER-OF-INITIAL-NODES
- Files contain the result of the CS / Ranking Execution
- Files additionally ending with "_all_community" contain the ids of the complete communities.

## Analysis
All primary data for the experiments like images, tables and data are stored in 07_analysis. 
- Files are named after their corresponding label in the thesis.
- Notebooks contain the produced labels in their title as well.