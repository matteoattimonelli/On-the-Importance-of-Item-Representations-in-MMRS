# Representation Choice Matters in Multimodal Recommendation: A Systematic Study on the Impact of LVLMs and Embedding-Oriented Item Representations


This repo contains the code to reproduce the results of the paper "_Representation Choice Matters in Multimodal Recommendation: A Systematic Study on the Impact of LVLMs and Embedding-Oriented Item Representations_", under review at ACM Transactions on Intelligent Systems and Technology (ACM TIST).

## Setting Up the Virtual Environment

After cloning the repository, it is recommended to create a virtual environment for installing the required dependencies. The codebase was developed and tested with **Python 3.11.6**, **CUDA 11.8**, and **Ubuntu 24.04.4 LTS**. All the experiments reported in the paper were executed on a machine equipped with an **NVIDIA A100 GPU**. 

To set up the virtual environment using `venv`, follow these steps:  

```sh
python3 -m venv venv
source venv/bin/activate
./install.sh
```

## Prompting
To generate structured metadata for images, we employ predefined prompts tailored to each dataset: **Baby, Pets, Clothing, Allrecipes, MovieLens**. As mentioned in the paper, we followed three prompting strategies: Fill-in-the-blank Prompt (FITB), Descriptive Prompt (DESCR), and Task-Misaligned Prompt (MIS).

### Fill-in-the-blank (FITB) Prompt

#### Baby Dataset

```
Imagine you’re creating metadata for an image database of baby products. Your task is to select five keywords that best represent the image content. Fill in the blanks: [Category], [Age Group], [Purpose], [Material], [Usage Context]. For example, your answer will be: [Category] {Feeding}, [Age Group] {Infant}, [Purpose] {Hygiene}, [Material] {Silicone}, [Usage Context] {Home}.
```

#### Pets Dataset

```
Imagine you’re creating metadata for an image database of pet-related items. Your task is to select five keywords that best represent the image content. Fill in the blanks: [Category], [Pet Type], [Purpose], [Material], [Usage Context]. For example, your answer will be: [Category] {Toys}, [Pet Type] {Dog}, [Purpose] {Entertainment}, [Material] {Rubber}, [Usage Context] {Outdoor}.
```

#### Clothing Dataset  

```
Imagine you're creating metadata for an image database of clothing items. Your task is to select five keywords that best represent the image content. Fill in the blanks: [Type], [Color], [Wear Location], [Material], [Style]. For example, your answer will be: [Type] {dress}, [Color] {red}, [Wear Location] {torso}, [Material] {cotton}, [Style] {casual}. 
```

#### Allrecipes Dataset

```
Imagine you’re creating metadata for a recipe recommendation system. Select five keywords that best represent the dish shown in the image. Fill in the blanks: [Dish Type], [Cuisine], [Primary Ingredient], [Flavor Profile], [Meal Type]. For example, your answer will be: [Dish Type] {Pasta} [Cuisine] {Italian} [Primary Ingredient] {Cheese} [Flavor Profile] {Savory} [Meal Type] {Dinner}.
```

#### MovieLens Dataset

```
Imagine you’re creating metadata for an image database of movie posters. Your task is to select five keywords that best represent the image content. Fill in the blanks: [Genre], [Audience], [Visual Format], [Mood], [Setting]. For example, your answer will be: [Genre] {Comedy}, [Audience] {Adults}, [Visual Format] {Live Action}, [Mood] {Witty}, [Setting] {Urban}.
```

### Descriptive Prompt (DESCR)

#### Baby Dataset

```
Imagine you’re creating metadata for an image database of baby products. Your task is to produce a comprehensive yet concise description that accurately represents the visual content of the image in 30 words at most.
```

#### Allrecipes Dataset

```
Imagine you’re creating metadata for arecipe recommendation system. Your task is to produce a comprehensive yet concise description that accurately represents the visual content of the image in 30 words at most.
```

#### MovieLens Dataset

```
Imagine you’re creating metadata for an image database of movie posters. Your task is to produce a comprehensive yet concise description that accurately represents the visual content of the image in 30 words at most.
```

### Task-Misaligned Prompt (MIS)

```
You are a computer scientist. Describe in fewer than 30 words how you can use this instrument for building a rocket.
```

### LVLMs Output Format

The final output from LVLMs is structured as a CSV file with the following format:
```
Image Name,Keywords
B0BHTBS5RM,"[Category] {Pet Accessories}, [Pet Type] {Dog}, [Purpose] {Support}, [Material] {Plastic}, [Usage Context] {Home}"
B0BX76YVP9,"[Category] {Leashes}, [Pet Type] {Dog}, [Purpose] {Training}, [Material] {Leather}, [Usage Context] {Outdoor}."
B0BM6V2SH8,"[Category] {Dog Treats}, [Pet Type] {Dog}, [Purpose] {Chewing}, [Material] {Meat}, [Usage Context] {Training Aid}"
31519,"[Dish Type] {Fries} [Cuisine] {American} [Primary Ingredient] {Potatoes} [Flavor Profile] {Spiced} [Meal Type] {Appetizer}"
1029,"[Genre] {Comedy}, [Audience] {Children}, [Visual Format] {Live Action}, [Mood] {Cheerful}, [Setting] {Circus}"
1120,"[Genre] {Comedy}, [Audience] {Adults}, [Visual Format] {Live Action}, [Mood] {Humorous}, [Setting] {Urban}"
1226,"[Genre] {Comedy}, [Audience] {Adults}, [Visual Format] {Illustration}, [Mood] {Cheerful}, [Setting] {Rural}"
2465,"[Genre] {Horror}, [Audience] {Teenagers/Adults}, [Visual Format] {Live Action}, [Mood] {Tense}, [Setting] {Indoor}"
2279,"[Genre] {Horror}, [Audience] {Teenagers/Young Adults}, [Visual Format] {Live Action}, [Mood] {Frightening}, [Setting] {Urban}"
```

```
Image Name,Description
2465,"A dark, moody poster for ""Deadly Friend,"" featuring a glowing, ominous figure with wings, set against a shadowy background, with the title in a distressed font."
2279,"A poster for ""Urban Legend"" features a central eye with multiple reflections of a young woman\'s face, symbolizing the film\'s theme of urban myths and paranoia."
3618,"A shadowy figure in a hat holds a pie, with the title 'SMALL TIME CROOKS' in bold red letters, suggesting a crime or detective theme."
9857,"White meringue cookies with red and green sprinkles, stacked on a dark surface."
44839,"A burger with a beef patty, melted cheese, lettuce, and sliced tomatoes, served in a soft, toasted bun."
```

```
Image Name,Description
B00A0ID90A,"This image shows a potty training seat, which is not useful for building a rocket."
B0BW5KRPSX,This image is not related to building a rocket. It shows a product for teething relief.
B07CR1MJ29,"This instrument appears to be a rocket igniter, which is used to ignite the fuel in a rocket engine."
B0BW8LFC6V,"I'm sorry, but I cannot provide assistance with that request."
B07GBTFRQZ,"This bag is not suitable for rocket construction; it's a children's toy. the bag is white with a pattern of lions on it the bag is white with a pattern of lions on it"
```

The csv outputs extracted using both LVLMs are available in the folder `extracted_outputs/`.

## Running the Experiments  

### Elliot

Available models:

- VBPR
- LATTICE
- FREEDOM
- BM3

To run the experiments, first download the required feature files from the following [link](https://drive.google.com/drive/folders/1ak4tJQdN1wTC2D_Vj09odDOiQVCwRIWp?usp=sharing).

Place each zip file into:

```
elliot/data/<dataset-name>/
```

Next, extract the features by executing the following commands:  

```sh
cd elliot/data/<dataset-name>
unzip <features.zip>
```

After extracting the features, navigate back to `elliot` directory and run:

```sh
export PYTHONPATH=.
```

To start the experiments, use the following command:

```sh
CUBLAS_WORKSPACE_CONFIG=:16:8 python3 start_experiments.py \
     --config <config_name> # config file basename
```

All necessary configuration files for reproducing our experiments are available in the `config_files` directory.

### MMRec

Available models:

- LGMRec
- PGL
- COHESION
- SMORE

Store features in:

```sh
data/<dataset-name>/visual_embeddings_indexed_32/ 
data/<dataset-name>/textual_embeddings_indexed_32/ 
```

Next, extract the features by executing the following commands:  

```sh
cd mmrec/data/<dataset-name>

# Visual features
cd visual_embeddings_indexed_32
unzip <visual_features.zip>

# Textual features
cd ../textual_embeddings_indexed_32
unzip <textual_features.zip>
```

Then, navigate back to `mmrec/src` directory of the project.

To start the experiments, use the following command:

```sh
cd mmrec/src
export PYTHONPATH=.

CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 run_MMRec_benchmarking.py \
        --data <dataset_name> \
        --model <model_name> \
        --visual_emb <visual_feature_dirname> \
        --textual_emb <textual_feature_dirname> \
        --extractor <extractor_name>
```

Use a dataset name available under `mmrec/data`, one of the listed model names, and the visual/textual embedding directory basenames to set the output extractor name.

Please note that for some MMRec models is is necessary to download also the 'user_graph_dict.npy' files, which can be found [here](https://drive.google.com/drive/folders/1NT-PK8xu-hU3ELIw9HUdlRipi0jkc9Yd?usp=sharing) or created using the script located under ```mmrec/preprocessing/dualgnn-gen-u-u-matrix.py```.

## The Team

Currently, this repository is maintained by:
- Matteo Attimonelli (matteo.attimonelli@poliba.it)
- Simone Bonfrate (s.bonfrate2@studenti.poliba.it)
- Danilo Danese (danilo.danese@poliba.it)
- Claudio Pomo (claudio.pomo@poliba.it)
- Antonio Ferrara (antonio.ferrara@poliba.it)
- Dietmar Jannach (Dietmar.Jannach@aau.at)
- Fedelucio Narducci (fedelucio.narducci@poliba.it)
- Tommaso Di Noia (tommaso.dinoia@poliba.it)
