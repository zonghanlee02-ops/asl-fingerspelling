# Colab notebooks for the sequence model

Upload these to [Google Colab](https://colab.research.google.com). They train the model; the live app only loads the exported model.

| Notebook | Purpose |
|---|---|
| **`fsboard_tcn_ctc_colab.ipynb`** | **Start here.** Does everything on FSboard: choose clips, get all 21 hand points from the videos, train, evaluate, export. Needs only itself. Rebuild it with `scripts\sequence_model\build_tcn_ctc_notebook.py` |
| `fsboard_prepare_data_colab.ipynb` | Older: inspects FSboard and exports its stored landmarks, which hold only 8 of 21 hand points. Needs only itself. Do not edit its hidden Setup cell: rebuild it with `scripts\sequence_model\build_fsboard_notebook.py` |
| `asl_seq_tcn_ctc_colab.ipynb` | Older: trains the TCN + CTC model on the earlier competition dataset, not FSboard. Needs the code zip from `scripts\sequence_model\seq_pack_code.py` |

Clear outputs before sharing: FSboard preview videos show signers' faces.
Details: [docs/SEQUENCE_MODEL.md](../docs/SEQUENCE_MODEL.md).
