"""
asl.seq - EXPERIMENTAL continuous-fingerspelling backend (TCN + CTC).

This package is separate from the baseline (asl/classifier.py and
baseline/model/), which it never imports or modifies.

    camera / dataset landmarks
      -> features.SequenceFeaturizer   68 numbers per frame, causal (no future frames)
      -> features.Normalizer            mean/std fitted on the TRAINING split only
      -> model.build_tcn_ctc            causal temporal convolution network
      -> ctc.greedy_decode / StreamingGreedyDecoder   -> ordered letters A-Z

Modules
-------
vocab      blank = 0, A-Z = 1..26, transcript normalisation and filtering
features   68-feature definition, causal featurizer, normaliser, resampler
ctc        CTC decoding (offline + streaming), edit distance, metrics
model      TCN-CTC architecture, receptive field, save/load of inference bundles
data       Google ASL Fingerspelling metadata, subset selection, extraction,
           signer split, manifests, batching, augmentation
train      explicit CTC training loop, resumable checkpoints
evaluate   evaluation report (CER, word accuracy, repeats, speed, idle)
inference  offline (Start/Finish) and streaming recognisers for local use
fsboard    Google's FSboard dataset: choosing clips, downloads, decoding the
           landmark records, exporting compact training examples

Only numpy is needed to import vocab/features/ctc. TensorFlow is imported
lazily by model/train/inference; pandas/pyarrow only by data (Colab side).
"""
