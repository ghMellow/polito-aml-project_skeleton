# Project Skeleton - Structured ML/DL Project Template

Template di progetto strutturato per Machine Learning/Deep Learning con PyTorch. Implementa best practices per organizzazione, riproducibilità e collaborazione.

**Forked from:** `iurada/project-skeleton:main`

---

## 📁 Struttura del Progetto

```
polito-aml-project_skeleton/
├── checkpoints/                  # 💾 MODEL CHECKPOINTS (created during training)
│   ├── .gitkeep                  # Mantiene cartella in git
│   ├── best_model.pth            # Best model salvato automaticamente (gitignored)
│   └── checkpoint_epoch_N.pth    # Checkpoint periodici (gitignored)
│
├── data/                         # 📁 DATASET FILES (gitignored - download separately)
│   ├── .gitkeep
│   ├── training_set/             # Training images (cats/, dogs/)
│   └── test_set/                 # Test images (cats/, dogs/)
|
├── dataset/                      # 📦 DATASET MODULE
│   ├── __init__.py               # Exports: CustomImageDataset, create_annotations_csv
│   └── custom_dataset.py         # PyTorch Dataset class per caricamento dati da CSV
│
├── models/                       # 🧠 MODELS MODULE
│   ├── __init__.py               # Exports: create_name_model
│   └── vgg_finetuning.py         # Architetture modelli
│
├── utils/                        # 🛠️ UTILITIES MODULE
│   ├── __init__.py               # Exports: transforms, visualization, metrics functions
|   ├── download_dataset.py       # DATASET DOWNLOADER (scarica dataset, es da Kaggle)
│   ├── transforms.py             # Data augmentation e preprocessing (train/val/test)
│   ├── visualization.py          # Plotting e visualizzazioni (denormalize, plot curves)
│   └── metrics.py                # Calcolo metriche e statistiche dataset
│
├── train.py                      # 🚂 TRAINING SCRIPT (main training loop con CLI)
├── eval.py                       # 📊 EVALUATION SCRIPT (test set evaluation con CLI)
├── config.py                     # ⚙️ CONFIGURATION (hyperparameters e settings centrali)
│
├── colab_training.ipynb          # 📓 GOOGLE COLAB NOTEBOOK (training su Colab)
├── requirements.txt              # 📋 PYTHON DEPENDENCIES (pip install -r requirements.txt)
├── .gitignore                    # 🚫 GIT IGNORE (data/, checkpoints/*.pth, *.csv, wandb/)
│
└── README.md                     
```

---

## 🎯 Best Practices Implementate

✅ **Modularità**: Codice diviso in moduli riutilizzabili

✅ **CLI Interface**: Argparse per tutti gli script
   - **Cos'è?** Command-Line Interface permette di eseguire gli script da terminale passando parametri come opzioni (es: `--epochs 10 --lr 0.001`)
   - **Vantaggi:** Non devi modificare il codice per ogni esperimento, tutti i parametri sono configurabili da riga di comando
   - **Implementazione:** Usa `argparse` in Python per definire tutti gli argomenti disponibili (data_dir, epochs, batch_size, learning rate, ecc.)

✅ **Reproducibility**: requirements.txt + config.py
   - **config.py** definisce valori di default e costanti del progetto
   - **CLI arguments** permettono di sovrascrivere i default senza modificare il codice
   - I due approcci sono complementari: config.py è il "centro di controllo", CLI dà flessibilità per esperimenti

✅ **Checkpoint Management**: Auto-save best model

✅ **Logging**: Wandb integration

✅ **Documentation**: Docstrings + README completo

✅ **Git-friendly**: .gitignore appropriato

✅ **Data Augmentation**: Solo su train, non su val/test

✅ **Separation of Concerns**: train.py vs eval.py

---

## 🔍 Per AI Assistants

**Questo progetto segue una struttura modulare standard:**

1. **Dataset Module** (`dataset/`): Gestione caricamento dati
2. **Models Module** (`models/`): Architetture e model creation
3. **Utils Module** (`utils/`): Transforms, visualization, metrics
4. **Training Script** (`train.py`): Main training loop con CLI
5. **Eval Script** (`eval.py`): Test set evaluation
6. **Config** (`config.py`): Centralized configuration

**Key Points:**
- Ogni modulo ha `__init__.py` con exports espliciti
- CLI scripts usano argparse
- Training (train/validate/test functions)
- Checkpoint management automatico
- Wandb integration opzionale ma completa
- Transforms: AUGMENTATION solo su train!

**Quando suggerire modifiche:**
- Aggiungere nuovi modelli → `models/new_model.py`
- Nuove metriche → `utils/metrics.py`
- Nuovi datasets → `dataset/new_dataset.py`
- Training modifications → `train.py` (maintain CLI style)

---

## 🤝 Contributing

Per adattare questo skeleton al tuo progetto:

1. **Dataset**: Modifica `dataset/custom_dataset.py` per il tuo formato
2. **Model**: Aggiungi la tua architettura in `models/`
3. **Config**: Aggiorna `config.py` con i tuoi parametri
4. **Training**: Modifica `train.py` se necessario (mantieni CLI)
5. **Update README**: Documenta le modifiche

---

## 🚫 Git Ignore (`.gitignore`)

**Cosa ignora:**
- `data/` - Dataset (troppo grande, download separato)
- `checkpoints/*.pth` - Model checkpoints (troppo grandi)
- `*.csv` - Annotation files (generati automaticamente)
- `wandb/` - Wandb logs (sincronizzati su cloud)
- `__pycache__/` - Python cache
- `.DS_Store` - macOS files

**Cosa traccia:**
- Codice sorgente (`.py`)
- Configurazioni
- README e docs
- `.gitkeep` per cartelle vuote

---

## 🔄 Workflow Tipico

### 1. Setup Iniziale
```bash
git clone <repo-url>
cd polito-aml-project_skeleton
pip install -r requirements.txt
python download_dataset.py
```

### 2. Training
```bash
# Feature extraction (base frozen)
python train.py --data_dir ./data --epochs 10 --freeze_base --use_wandb

# Full fine-tuning (tutto trainable)
python train.py --data_dir ./data --epochs 10 --use_wandb
```

### 3. Evaluation
```bash
python eval.py --checkpoint ./checkpoints/best_model.pth --data_dir ./data
```

### 4. Experiments
```bash
# Esperimento con LR diverso
python train.py --lr 0.001 --batch_size 64 --use_wandb

# Tutti gli esperimenti tracciati su Wandb!
```

---

## 📢 Informazioni di rilascio

**📅 Ultimo aggiornamento:** Novembre 2025  
**🏷️ Versione:** v1.0.0 — Prima release stabile

*Per dettagli sui cambiamenti e le correzioni, consulta il changelog nel repository.*
