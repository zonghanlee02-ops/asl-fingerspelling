# Datasets

Nothing here is published to GitHub except this note and the word lists.

| Path | What it is |
|---|---|
| `my_words_train.txt` | 100 words you record to fine-tune the sequence model on your own signing (Phase 2A) |
| `my_words_test.txt` | 40 different words you record only for testing; never used for training |
| `my_phrases_train.txt` | 49 short phrases with spaces, to teach how you mark the gap between words (Phase 2B) |
| `my_phrases_test.txt` | 14 different phrases, only for testing spaces |
| `my_doubles_train.txt` | 21 extra words with double letters (BOOK, LETTER), the main remaining error |
| `my_i_words_train.txt` | 14 words that start with I, including I alone, because the model missed the I in I LOVE YOU |
| `words_en.txt` | 30,000 common English words with counts, for word correction (Phase 3). Made by `scripts/sequence_model/build_word_list.py` from norvig.com/ngrams/count_1w.txt |
| `word_pairs_en.txt` | the 100,000 most common two-word sequences of those words with counts (2 MB), for the word decoder (stage 2). Made by `scripts/sequence_model/build_word_pairs.py` from norvig.com/ngrams/count_2w.txt; check its licence before publishing |
| `my_extra_words.txt` | your own names and words that word correction should know |

Your recordings of these words are personal data and stay in `backend/storage/`, which git ignores.

Large datasets:

| Path | What it is | How to get it back |
|---|---|---|
| `reference_poses_kaggle_backup.json` | the 3D reference poses before 9 letters were rebuilt from your takes (23 Nov 2026) | Copy letters back into `frontend/public/reference_poses.json` to undo |
| `landmarks.npz` | hand points from the ASL Alphabet photos; the source of the 3D reference poses | Its extractor was removed with the baseline model (23 Nov 2026), so keep this file |
| `fsboard_pilot/` | FSboard export, if you saved one from Colab | `notebooks/fsboard_prepare_data_colab.ipynb` |

The ASL Alphabet photos ([Kaggle](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)) were removed with
the baseline model on 23 Nov 2026; nothing in the project needs them now.
