# Baselines for CDEC-WN

We provide two baselines for the CDEC-WN dataset, the standard lemma-match baseline and a BERT-based cross-encoder. Refer to our paper for a discussion on the baseline performance and potential directions for future work.

## Requirements

Install the below requirements,

* pytorch (tested with 1.8.1)
* huggingface transformers (tested with 4.6.1)
* spacy (tested with 3.1.3)
* scikit-learn (tested with 0.21.3)
* tqdm

## Preprocess

Use the subtopic splits from `dataset_splits` to prepare training data for cross-validation (k=5).

```bash
# Download the dataset to the root directory
wget https://github.com/adithya7/cdec-wikinews/releases/download/v1.0/cdec-wn-dataset.tar.gz -P ../
tar -zxvf ../cdec-wn-dataset.tar.gz -C ../
```

```bash
# prepare training data
RAW_DATA=../cdec-wn-dataset
DATA=data_155/
python preprocess.py \
    -coref_pairs ${RAW_DATA}/dataset_labels/coref_pairs.json \
    -docs ${RAW_DATA}/dataset_docs/ \
    -splits ${RAW_DATA}/dataset_splits/ \
    -out_dir ${DATA}/
```

## Run lemma match baseline

Run the standard lemma-match baseline.

```bash
RAW_DATA=../cdec-wn-dataset
DATA=data_155/
for i in `seq 0 4`;
do
    echo "----------------------"
    echo "cross-validation: k="$i
    python lemma_baseline.py \
        -docs ${RAW_DATA}/dataset_docs/ \
        -subtopics ${DATA}/dev_${i}_subtopics.txt \
        -dev_path ${DATA}/dev_${i}.json \
        -sim_threshold 0
    echo "----------------------"
done

python lemma_baseline.py \
    -docs ${RAW_DATA}/dataset_docs/ \
    -subtopics ${RAW_DATA}/dataset_splits/test_subtopics.txt \
    -dev_path ${DATA}/test_pairs.json \
    -sim_threshold 0
```

## Run cross-encoder baseline

Run the cross-encoder baseline.

```bash
DATA=data_155/
mkdir -p logs
for i in `seq 0 4`;
do
    echo "----------------------"
    echo "cross-validation: k="$i
    python train.py train \
        -train_path ${DATA}/train_${i}.json \
        -dev_path ${DATA}/dev_${i}.json \
        -save_dir saved_models/ \
        -config configs/config_event_tag.json
    echo "----------------------"
done
```

## License

Code for the baselines is available under MIT License.
