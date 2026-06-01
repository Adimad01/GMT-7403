# Expérimentations — GPT-OSS 20B + Raisonnement Topologique par KG OSM

## Vue d'ensemble

Ce dossier contient 5 expérimentations qui évaluent la capacité du modèle **GPT-OSS-20B** à prédire les relations topologiques DE-9IM (`contains`, `within`, `touches`, `crosses`, `disjoint`, `overlaps`) entre entités géographiques.

Chaque expérimentation est évaluée sur **96 exemples équilibrés** (16 par prédicat) avec trois stratégies de raisonnement : **CoT**, **ToT**, **GoT**.

---

## Architecture du code

```
eval_engine_gpu.py       ← moteur principal d'inférence GPU
strategies_osm.py         ← implémentation CoT / ToT / GoT + accès KG OSM
train_lora_adapter.py        ← script de fine-tuning LoRA
analyze_experiments.py ← analyse et visualisation des 5 × 3 résultats
exp01_base_model.py       ← Expérimentation 1
exp02_finetuned_topo.py  ← Expérimentation 2
exp03_finetuned_osm_kg.py    ← Expérimentation 3
exp05_finetuned_extended.py ← Expérimentation 4
exp06_base_ollama.py       ← Expérimentation 5
```

---

## Stratégies de raisonnement (`strategies_osm.py`)

Chaque stratégie reçoit une paire de lieux (A, B) et construit un prompt enrichi avec les données OSM (coordonnées, bounding box, hiérarchie administrative) avant d'appeler le LLM.

### `GeographicKnowledgeGraph`
Classe principale d'accès au KG OSM.
- `get_neighborhood_details(place)` — interroge Nominatim pour récupérer : coordonnées, type de lieu, bounding box, hiérarchie administrative. Résultats mis en cache dans `results/osm_cache.json`.

### `CoTStrategy` (Chain-of-Thought)
Construit un unique prompt avec toutes les preuves OSM et demande au modèle de raisonner étape par étape avant de donner un prédicat. Une seule génération par exemple.

### `ToTStrategy` (Tree-of-Thought)
Génère plusieurs hypothèses de prédicats (branches), évalue chacune avec un prompt distinct, puis sélectionne la meilleure via un vote ou score de confiance. Plusieurs appels LLM par exemple.

### `GoTStrategy` (Graph-of-Thought)
Décompose le raisonnement en étapes interdépendantes (nœuds d'un graphe) : extraction des preuves → analyse géométrique → comparaison → conclusion. Chaque nœud est une génération LLM distincte.

---

## Moteur d'inférence GPU (`eval_engine_gpu.py`)

Script central appelé par tous les scripts d'expérimentation via `sys.argv` + `from eval_engine_gpu import main`.

### Patches appliqués au démarrage
| Patch | Raison |
|-------|--------|
| `_TorchvisionStubFinder` | torchvision installé mais cassé sur ce serveur (`_cast_Long` manquant). Un `MetaPathFinder` intercepte tous les imports `torchvision.*` et retourne des modules vides. |
| `torch.accelerator` | Manquant dans PyTorch < 2.4. Polyfill ajouté avant l'import de transformers. |
| `torch.nn.Module.set_submodule` | Manquant dans PyTorch < 2.5. Polyfill ajouté. |
| `AutoHfQuantizer.supports_quant_method` | Bug transformers ≥5.9 : crash quand `quantization_config=None`. La méthode est remplacée par une version sûre. |

### Fonctions principales

**`_save_json_atomic(path, data)`**
Sauvegarde atomique du checkpoint JSON via un fichier temporaire + `os.replace`. Évite la corruption du checkpoint si le processus est tué en cours d'écriture.

**`_load_checkpoint(ckpt_path)`**
Charge un checkpoint existant (indices traités + résultats). Retourne un dict vide si le fichier n'existe pas ou est corrompu. Permet la **reprise automatique** des expérimentations interrompues.

**`evaluate_strategy(strategy, df, output_dir, model_tag)`**
Boucle d'évaluation principale :
1. Charge le checkpoint pour reprendre là où on s'était arrêté.
2. Pour chaque exemple non encore traité : appelle la stratégie → récupère le prédicat prédit → compare avec la vérité terrain.
3. Sauvegarde le checkpoint après chaque exemple.
4. Affiche la précision en temps réel dans la barre `tqdm`.

**`gpu_inference_fn(prompt)`**
Fonction d'inférence locale GPU passée aux stratégies à la place d'Ollama :
- Tokenise le prompt avec troncature à `max_position_embeddings - max_new_tokens`.
- Génère avec `model.generate()` (sampling, température 0.1).
- Décode uniquement les tokens générés (pas le prompt).
- Logue la latence du premier exemple.

### Chargement du modèle
1. Vérifie CUDA disponible → affiche le GPU et la mémoire.
2. Charge GPT-OSS-20B en `bfloat16` sans quantification (le modèle tient dans 80 GB).
3. Si un `adapter_path` est fourni, applique l'adaptateur LoRA via `PeftModel.from_pretrained`. Un patch intercepte `hf_hub_download` pour permettre les chemins absolus locaux.

---

## Script de fine-tuning (`train_lora_adapter.py`)

Utilisé pour créer les adaptateurs LoRA.

### Fonction `build_training_prompt(row, tokenizer)`
Construit un exemple d'entraînement complet au format instruction-following :
- **Système** : rôle d'expert en raisonnement topologique DE-9IM.
- **Entrée** : relation vernaculaire + types géométriques de A et B.
- **Sortie** : raisonnement + prédicat DE-9IM attendu.

### Fonction `main()`
1. Charge le dataset CSV et le tokenizer.
2. Construit le `Dataset` HuggingFace à partir des exemples.
3. Charge GPT-OSS-20B en bfloat16.
4. Attache les adaptateurs LoRA (`r=8`, `lora_alpha=16`, cibles : `q_proj`, `v_proj`, `k_proj`, `o_proj`).
5. Lance `SFTTrainer` (TRL) — 3 epochs, batch=1, gradient accumulation=16, lr=2e-4.
6. Sauvegarde l'adaptateur dans `{output_dir}/final_adapter/`.

---

## Expérimentation 1 — GPTOSS Base (`exp01_base_model.py`)

**Objectif** : établir une ligne de base sans fine-tuning ni adaptateur.

**Configuration** :
- Modèle : `openai/gpt-oss-20b` (pas d'adaptateur)
- Budget tokens : 512
- KG : OSM (Nominatim)

**Résultats** :
| Stratégie | Précision |
|-----------|-----------|
| CoT | 57.3% |
| ToT | 37.5% |
| GoT | 26.0% |

**Logique du script** :
- `preflight()` : vérifie que le dataset et le fichier d'indices existent.
- `check_strategy_status()` : lit les checkpoints et affiche l'état de chaque stratégie (COMPLETE / PARTIAL / NOT STARTED).
- `run()` : configure `sys.argv` puis appelle `eval_engine_gpu.main()` sans passer d'`adapter_path`.

---

## Expérimentation 2 — GPTOSS Fine-tuné (`exp02_finetuned_topo.py`)

**Objectif** : évaluer l'apport du fine-tuning sur données topologiques brutes (sans KG en entraînement).

**Configuration** :
- Modèle : `openai/gpt-oss-20b` + adaptateur `finetuned_gptoss_topological/final_adapter`
- Entraîné sur : `triplet_update_v3_70.csv` (755 exemples, relation + types géométriques uniquement)
- Budget tokens : 512
- KG à l'inférence : OSM

**Résultats** :
| Stratégie | Précision |
|-----------|-----------|
| CoT | 53.1% |
| ToT | 19.8% |
| GoT | **61.5%** ← meilleure |

**Logique du script** : identique à Exp1 mais passe `ADAPTER_PATH = "finetuned_gptoss_topological/final_adapter"` à `eval_engine_gpu`.

---

## Expérimentation 3 — GPTOSS Fine-tuné + KG en entrée (`exp03_finetuned_osm_kg.py`)

**Objectif** : évaluer un modèle fine-tuné *avec* KG OSM en entraînement ET à l'inférence.

**Configuration** :
- Modèle : `openai/gpt-oss-20b` + adaptateur `finetuned_gptoss_osm_kg/final_adapter`
- Entraîné sur : `osm_kg_train.jsonl` (exemples avec preuves OSM intégrées dans le prompt)
- Budget tokens : 1024 (budget étendu car prompts plus longs avec KG)
- KG à l'inférence : OSM

**Résultats** :
| Stratégie | Précision |
|-----------|-----------|
| CoT | **47.9%** ← meilleure |
| ToT | 24.0% |
| GoT | 27.1% |

**Logique du script** : passe `ADAPTER_PATH = "finetuned_gptoss_osm_kg/final_adapter"` et `MAX_NEW_TOKENS = 1024`.

---

## Expérimentation 4 — GPTOSS Fine-tuné + Inférence LLM enrichie par KG (`exp05_finetuned_extended.py`)

**Objectif** : tester si un budget de raisonnement plus large (1024 vs 512 tokens) améliore un modèle fine-tuné sur données brutes.

**Configuration** :
- Modèle : `openai/gpt-oss-20b` + adaptateur `finetuned_gptoss_topological/final_adapter` (même que Exp2)
- Budget tokens : **1024** (double de Exp2)
- KG à l'inférence : OSM

**Différence clé avec Exp2** : le modèle est identique, seul le budget de génération change. Teste si le modèle exploite mieux un espace de raisonnement plus large.

**Résultats** :
| Stratégie | Précision |
|-----------|-----------|
| CoT | 49.0% |
| ToT | 29.2% |
| GoT | **56.2%** ← meilleure |

**Logique du script** : même adaptateur qu'Exp2, mais `MAX_NEW_TOKENS = 1024`.

---

## Expérimentation 5 — GPTOSS + Inférence LLM enrichie par KG (`exp06_base_ollama.py`)

**Objectif** : ligne de base avec inférence via Ollama (endpoint distant) et raisonnement enrichi par KG OSM.

**Configuration** :
- Modèle : `gpt-oss` base via Ollama (`http://ollama.apps.crdig.ulaval.ca`)
- Pas d'adaptateur
- Budget tokens : 1024
- KG à l'inférence : OSM

**Particularité** : contrairement aux expérimentations 1–4 qui utilisent `eval_engine_gpu.py` (inférence GPU locale), cette expérimentation appelle le modèle via l'API Ollama distante. La fonction `model_fn` est passée à `get_strategy()` dans `strategies_osm.py`.

**Résultats** :
| Stratégie | Précision |
|-----------|-----------|
| CoT | 66.7% |
| ToT | 69.8% |
| GoT | **74.0%** ← meilleure de toutes les expérimentations |

---

## Analyse et visualisation (`analyze_experiments.py`)

### `load_ckpt(ckpt_path)`
Charge un fichier checkpoint JSON et retourne un DataFrame pandas avec colonnes `index`, `expected`, `predicted`, `match`.

### `compute_metrics(df, label)`
Calcule :
- Précision globale
- Précision par prédicat DE-9IM (contains, within, touches, crosses, disjoint, overlaps)

### `print_table(results_matrix)`
Affiche un tableau de comparaison par stratégie avec précisions globales et par prédicat.

### `plot_confusion_matrix(df, title, save_path)`
Génère une matrice de confusion (prédit vs attendu) pour chaque expérimentation × stratégie. Sauvegardée en PNG dans `results/`.

### `plot_grouped_bar(results_matrix, save_path)`
Génère un graphique à barres groupées comparant toutes les expérimentations × stratégies. Sauvegardé dans `results/acc_96_experiments_by_strategy.png`.

---

## Résumé des résultats

| Expérimentation | Meilleure stratégie | Précision |
|-----------------|--------------------:|-----------|
| GPTOSS Base | CoT | 57.3% |
| GPTOSS Fine-tuné | GoT | 61.5% |
| GPTOSS Fine-tuné + KG en entrée | CoT | 47.9% |
| GPTOSS Fine-tuné + Inférence enrichie/KG | GoT | 56.2% |
| **GPTOSS + Inférence LLM enrichie par KG** | **GoT** | **74.0%** |

**Observation principale** : le modèle de base sans fine-tuning mais avec une stratégie GoT enrichie par KG OSM (via Ollama) surpasse tous les modèles fine-tunés. Le fine-tuning seul n'améliore pas systématiquement les résultats ; c'est la combinaison de la stratégie de raisonnement (GoT) et du KG qui fait la différence.
