# Rapport : Approche Prompt-Based pour le Raisonnement Topologique

**Auteur** : Imad Eddine Lassakeur  
**Date** : 13 avril 2026  
**Projet** : Topological Reasoning — Conversion de descriptions spatiales vernaculaires en prédicats topologiques DE-9IM

---

## 1. Objectif

Convertir automatiquement des descriptions spatiales en langage naturel (vernaculaire) en prédicats topologiques formels du modèle **DE-9IM** (Dimensionally Extended 9-Intersection Model), en utilisant un LLM (Large Language Model) guidé par trois stratégies de raisonnement avancées.

**Exemple** :  
- *Entrée* : `"Palo Alto is adjacent to Stanford University"`  
- *Sortie attendue* : `touches`

Les **7 prédicats cibles** sont : `within`, `contains`, `touches`, `crosses`, `disjoint`, `overlaps`, `equals`.

---

## 2. Deux modes de construction du Knowledge Graph

Le pipeline repose sur un **Knowledge Graph (KG)** qui structure les informations géographiques avant le raisonnement. Deux modes de construction du KG ont été implémentés et évalués :

### Mode 1 : Tool-Based (OSM)
Le KG est construit via des **appels aux APIs OpenStreetMap** (Nominatim pour le géocodage, Overpass pour les requêtes spatiales). Pour chaque paire d'entités, le système récupère :
- La hiérarchie administrative (ville → comté → état → pays)
- Les lieux proches et leurs relations spatiales
- Les métadonnées géographiques (type, coordonnées, frontières)

**Avantage** : Données géographiques réelles et détaillées.  
**Inconvénient** : Appels API lents (~30s/ligne), données volumineuses qui « noient » le signal utile dans les prompts multi-branches, et dépendance à la disponibilité des APIs.

### Mode 2 : Prompt-Based (sans appel externe)
Le KG est construit **directement à partir des métadonnées du CSV** (noms, types de lieux, types de géométrie, relation vernaculaire). Aucun appel LLM ni API externe n'est nécessaire pour la construction du KG.

**Avantage** : Rapide (~3s/ligne), prompts concis et ciblés, zéro dépendance externe.  
**Inconvénient** : Pas de données géographiques enrichies (pas de hiérarchie administrative ni de contexte spatial réel).

Les deux modes utilisent les **mêmes trois stratégies de raisonnement** (CoT, ToT, GoT) et le **même modèle LLM** (`gpt-oss`). Seule la source des données du KG diffère.

---

## 3. Dataset

- **Fichier** : `triplet_update_v3_30.csv` — **323 lignes**
- **Colonnes** : `place_name_subject`, `place_name_object`, `placetype_subject/object`, `geometry_type_subject/object`, `Sentence`, `vernacular_relation`, `relation_predicate`, `spatial_relation` (ground truth)

### Distribution des classes

| Prédicat | Nombre | % |
|----------|--------|---|
| `touches` | 95 | 29.4% |
| `within` | 80 | 24.8% |
| `disjoint` | 79 | 24.5% |
| `overlaps` | 36 | 11.1% |
| `crosses` | 17 | 5.3% |
| `contains` | 16 | 5.0% |
| `equals` | 0 | 0.0% |

Le dataset est déséquilibré, avec `touches`, `within`, et `disjoint` dominant (~79% des cas).

---

## 4. Architecture de l'approche

### 4.1. Modèle LLM

- **Modèle** : `gpt-oss` (20.9B paramètres, quantification MXFP4)
- **Serveur** : Ollama hébergé à `http://ollama.apps.crdig.ulaval.ca`
- **Température** : 0.2 (favorise la cohérence des réponses)
- **Framework** : `langchain-ollama` (ChatOllama)

### 4.2. Construction du Knowledge Graph (KG) — Mode Prompt-Based

**Principe clé** : Le KG est construit **directement à partir des données du CSV** (zéro appel LLM pour la construction du KG).

#### Étape 1 : Extraction des données de la ligne CSV

Pour chaque ligne du dataset, un dictionnaire `entity` est construit avec les colonnes du CSV :

```python
entity = {
    "place_name_subject": str(row["place_name_subject"]),      # ex. "Baraboo, Wisconsin"
    "place_name_object":  str(row["place_name_object"]),       # ex. "Circus World Museum, Wisconsin"
    "placetype_subject":  str(row["placetype_subject"]),       # ex. "city"
    "placetype_object":   str(row["placetype_object"]),        # ex. "tourism"
    "geometry_type_subject": str(row["geometry_type_subject"]),# ex. "MultiPolygon"
    "geometry_type_object":  str(row["geometry_type_object"]), # ex. "MultiPolygon"
    "relation_predicate": str(row["relation_predicate"]),      # ex. "is home to"
    "sentence": str(row["Sentence"]),                          # phrase complète
}
```

#### Étape 2 : Construction des nœuds du KG

Deux nœuds `KGNode` sont créés directement (sans appel LLM ni API) :

```python
KGNode(name="Baraboo, Wisconsin", node_type="city", properties={"geometry_type": "MultiPolygon"})
KGNode(name="Circus World Museum, Wisconsin", node_type="tourism", properties={"geometry_type": "MultiPolygon"})
```

#### Étape 3 : Construction de l'arête (triplet)

Une arête `KGEdge` relie les deux nœuds avec la relation vernaculaire extraite du CSV :

```python
KGEdge(
    head="Baraboo, Wisconsin",           head_type="city",
    relation="is home to",               # relation vernaculaire brute du CSV
    tail="Circus World Museum, Wisconsin", tail_type="tourism"
)
```

#### Étape 4 : Formatage du bloc d'évidence pour le prompt

La méthode `gather_evidence()` produit le bloc de texte suivant, injecté dans le prompt de raisonnement :

```
Knowledge graph triple (from dataset):
  Baraboo, Wisconsin (city, geometry=MultiPolygon) --[is home to]--> 
      Circus World Museum, Wisconsin (tourism, geometry=MultiPolygon)
Sentence: "Baraboo is home to the Circus World Museum"
```

#### Résumé du pipeline KG

```
CSV row → entity dict → KGNode (×2) + KGEdge (×1) → evidence text → injecté dans le prompt LLM
```

| Caractéristique | Valeur |
|-----------------|--------|
| Appels LLM pour le KG | **0** |
| Appels API externes | **0** |
| Temps moyen par ligne | **~3 secondes** (uniquement le raisonnement LLM) |
| Données utilisées | Noms, types de lieux, types de géométrie, relation vernaculaire, phrase |

Cette approche est très efficace car elle élimine toute extraction LLM pour le KG. Le LLM n'intervient **que** pour le raisonnement topologique (CoT/ToT/GoT), jamais pour la construction du graphe.

### 4.3. Prompt Engineering

Chaque stratégie utilise un prompt structuré contenant :

#### a) Lexique vernaculaire (Vernacular Lexicon)
Un mini-dictionnaire avec un seul exemple par prédicat pour guider l'interprétation :
```
WITHIN    — e.g. "is in"          (A is inside B)
CONTAINS  — e.g. "is home to"     (A encloses B)
TOUCHES   — e.g. "borders"        (A and B share a boundary, no overlap)
CROSSES   — e.g. "passes through" (A traverses B)
OVERLAPS  — e.g. "overlaps with"  (A and B partially share area)
DISJOINT  — e.g. "is far from"    (A and B are completely separate)
EQUALS    — e.g. "is the same as" (A and B occupy the same space)
```

#### b) Règles de raisonnement (Rules Block)
```
1. La relation est DIRIGÉE : A [prédicat] B
2. Considérer les types de géométrie (Point, LineString, Polygon, MultiPolygon)
3. Choisir EXACTEMENT UN prédicat
4. Interpréter soigneusement l'expression vernaculaire
5. Considérer les types de lieux et de géométries
6. Utiliser les preuves du KG
7. Terminer par : Answer: [predicate]
```

#### c) Bloc contextuel (Entity Context)
```
Entity A: [nom] (type: [type], geometry: [geom])
Entity B: [nom] (type: [type], geometry: [geom])
Vernacular description: "[A] [relation] [B]"
Valid predicates: contains, within, touches, crosses, disjoint, overlaps, equals
```

#### d) Preuves du KG (KG Evidence)
Le triplet construit à partir des données de la ligne CSV.

---

## 5. Stratégies de raisonnement

### 5.1. Chain-of-Thought (CoT)

**Principe** : Raisonnement linéaire en 5 étapes séquentielles.

```
Étape 1 — ANALYSE LINGUISTIQUE : Que signifie l'expression vernaculaire ?
Étape 2 — RAISONNEMENT PAR TYPE : Quelles relations sont typiques entre ces types de lieux ?
Étape 3 — CONTRAINTES GÉOMÉTRIQUES : Quels prédicats sont possibles pour ces géométries ?
Étape 4 — ANALYSE DES PREUVES KG : Que disent les triplets extraits ?
Étape 5 — SYNTHÈSE & DÉCISION : Quel prédicat unique formalise le mieux la relation ?
```

**Appels LLM** : 1 par ligne (+ 1 fallback si aucun prédicat extrait)

### 5.2. Tree-of-Thought (ToT)

**Principe** : Exploration de 3 branches de raisonnement distinctes avec vote majoritaire.

```
Phase 1 — GÉNÉRATION : Le LLM génère 3 branches de raisonnement indépendantes,
           chacune avec une perspective différente. Chaque branche propose un prédicat.
Phase 2 — VOTE : Vote majoritaire parmi les 3 prédictions.
           En cas d'égalité → prompt de départage (tie-break).
```

**Appels LLM** : 1 (génération des 3 branches) + 0-1 (tie-break si nécessaire)

### 5.3. Graph-of-Thought (GoT)

**Principe** : Raisonnement structuré en graphe avec 4 nœuds de pensée, fusion pairwise, et agrégation finale.

```
Phase 1 — DÉCOMPOSITION : Le LLM génère 4 nœuds de pensée (Thought Nodes),
           chacun analysant la relation sous un angle différent.
Phase 2 — FUSION PAIRWISE : Les nœuds sont fusionnés par paires (1+2, 3+4),
           synthétisant les perspectives convergentes/divergentes.
Phase 3 — AGRÉGATION FINALE : Les nœuds fusionnés sont combinés pour
           produire une prédiction finale unique.
Fallback — Si aucun prédicat valide : vote majoritaire sur tous les nœuds.
```

**Appels LLM** : 1 (génération) + 2 (fusions pairwise) + 1 (agrégation finale) = **4 par ligne**

### 5.4. Prompts complets utilisés (Prompt-Based)

Voici les prompts exacts envoyés au LLM pour chaque stratégie. Les variables entre accolades (`{...}`) sont remplacées dynamiquement par les données de chaque ligne du CSV.

#### a) Prompt CoT (Chain-of-Thought)

```
You are an expert in geospatial topological reasoning.

Your task: Interpret the vernacular (everyday language) spatial description below
and convert it into exactly one formal topological predicate from the DE-9IM model.

Vernacular-to-Topology Reference (one example each):
  WITHIN    — e.g. "is in"         (A is inside B)
  CONTAINS  — e.g. "is home to"    (A encloses B)
  TOUCHES   — e.g. "borders"       (A and B share a boundary, no overlap)
  CROSSES   — e.g. "passes through"(A traverses B)
  OVERLAPS  — e.g. "overlaps with" (A and B partially share area)
  DISJOINT  — e.g. "is far from"   (A and B are completely separate)
  EQUALS    — e.g. "is the same as"(A and B occupy the same space)

Note: geometry types constrain possible relations.

Rules:
1. The relation is DIRECTED: A [predicate] B.
2. Consider geometry types (Point, LineString, Polygon, MultiPolygon).
3. Pick EXACTLY ONE predicate from: contains, within, touches, crosses, disjoint, overlaps, equals.
4. Carefully interpret the vernacular expression.
5. Consider what makes sense given the place types and geometry types involved.
6. Use the knowledge graph evidence to support your reasoning.
7. End with: Answer: [predicate]

Entity A: {place_name_subject} (type: {placetype_subject}, geometry: {geometry_type_subject})
Entity B: {place_name_object} (type: {placetype_object}, geometry: {geometry_type_object})
Vernacular description: "{place_name_subject} {relation_predicate} {place_name_object}"
Valid predicates: contains, within, touches, crosses, disjoint, overlaps, equals

--- KNOWLEDGE GRAPH EVIDENCE (LLM-extracted) ---
{kg_evidence}

Think step-by-step:

Step 1 — LANGUAGE ANALYSIS:
What does the expression "{relation_predicate}" mean in everyday language?
Which spatial concept does it convey (containment, adjacency, crossing, separation, overlap, equivalence)?

Step 2 — PLACE TYPE REASONING:
A is a {placetype_subject} and B is a {placetype_object}.
What topological relations typically hold between these types of places?

Step 3 — GEOMETRY CONSTRAINTS:
A has geometry {geometry_type_subject} and B has geometry {geometry_type_object}.
Which predicates are geometrically possible for this combination?

Step 4 — KG EVIDENCE ANALYSIS:
What do the extracted knowledge graph triples, hierarchies, and place info tell you?
Does this confirm or change your interpretation?

Step 5 — SYNTHESIS & DECISION:
Combining all the above, which single predicate best formalizes "{relation_predicate}"?

Reasoning:
```

#### b) Prompt ToT (Tree-of-Thought)

```
You are an expert in geospatial topological reasoning.

Vernacular-to-Topology Reference (one example each):
  WITHIN    — e.g. "is in"         (A is inside B)
  CONTAINS  — e.g. "is home to"    (A encloses B)
  TOUCHES   — e.g. "borders"       (A and B share a boundary, no overlap)
  CROSSES   — e.g. "passes through"(A traverses B)
  OVERLAPS  — e.g. "overlaps with" (A and B partially share area)
  DISJOINT  — e.g. "is far from"   (A and B are completely separate)
  EQUALS    — e.g. "is the same as"(A and B occupy the same space)

Note: geometry types constrain possible relations.

Rules:
1. The relation is DIRECTED: A [predicate] B.
2. Consider geometry types (Point, LineString, Polygon, MultiPolygon).
3. Pick EXACTLY ONE predicate from: contains, within, touches, crosses, disjoint, overlaps, equals.
4. Carefully interpret the vernacular expression.
5. Consider what makes sense given the place types and geometry types involved.
6. Use the knowledge graph evidence to support your reasoning.
7. End with: Answer: [predicate]

Entity A: {place_name_subject} (type: {placetype_subject}, geometry: {geometry_type_subject})
Entity B: {place_name_object} (type: {placetype_object}, geometry: {geometry_type_object})
Vernacular description: "{place_name_subject} {relation_predicate} {place_name_object}"
Valid predicates: contains, within, touches, crosses, disjoint, overlaps, equals

--- KNOWLEDGE GRAPH EVIDENCE (LLM-extracted) ---
{kg_evidence}

Using the evidence above, explore THREE different reasoning branches to determine
the topological predicate for: "{place_name_subject} {relation_predicate} {place_name_object}"

For each branch, take a DIFFERENT perspective or approach.
Each branch must end with a predicate suggestion.

Format your response EXACTLY as:

BRANCH 1: [title of your approach]
[your reasoning using the KG evidence]
Suggested predicate: [predicate]

BRANCH 2: [title of your approach]
[your reasoning using the KG evidence]
Suggested predicate: [predicate]

BRANCH 3: [title of your approach]
[your reasoning using the KG evidence]
Suggested predicate: [predicate]

Begin:
```

**Prompt de départage (Tie-Break)** — utilisé uniquement si les 3 branches donnent un vote ex æquo :
```
{context}
{kg_evidence}
You explored multiple reasoning branches and got: {votes}
There is a tie between: {tie_preds}

Given the expression "{relation_predicate}",
A is a {placetype_subject} ({geometry_type_subject}), B is a {placetype_object} ({geometry_type_object}):
Which predicate is most correct and why?

Answer: [
```

#### c) Prompt GoT (Graph-of-Thought)

**Phase 1 — Génération des 4 nœuds de pensée :**
```
You are an expert in geospatial topological reasoning.

Vernacular-to-Topology Reference (one example each):
  WITHIN    — e.g. "is in"         (A is inside B)
  CONTAINS  — e.g. "is home to"    (A encloses B)
  TOUCHES   — e.g. "borders"       (A and B share a boundary, no overlap)
  CROSSES   — e.g. "passes through"(A traverses B)
  OVERLAPS  — e.g. "overlaps with" (A and B partially share area)
  DISJOINT  — e.g. "is far from"   (A and B are completely separate)
  EQUALS    — e.g. "is the same as"(A and B occupy the same space)

Note: geometry types constrain possible relations.

Rules:
1. The relation is DIRECTED: A [predicate] B.
2. Consider geometry types (Point, LineString, Polygon, MultiPolygon).
3. Pick EXACTLY ONE predicate from: contains, within, touches, crosses, disjoint, overlaps, equals.
4. Carefully interpret the vernacular expression.
5. Consider what makes sense given the place types and geometry types involved.
6. Use the knowledge graph evidence to support your reasoning.
7. End with: Answer: [predicate]

Entity A: {place_name_subject} (type: {placetype_subject}, geometry: {geometry_type_subject})
Entity B: {place_name_object} (type: {placetype_object}, geometry: {geometry_type_object})
Vernacular description: "{place_name_subject} {relation_predicate} {place_name_object}"
Valid predicates: contains, within, touches, crosses, disjoint, overlaps, equals

--- KNOWLEDGE GRAPH EVIDENCE (LLM-extracted) ---
{kg_evidence}

Generate FOUR distinct thought nodes, each analyzing the expression
"{place_name_subject} {relation_predicate} {place_name_object}" from a DIFFERENT angle.
Use the KG evidence to ground each thought.

Format EXACTLY as:

THOUGHT 1: [your angle/approach]
[reasoning from this angle, using KG evidence]
Predicate: [predicate]

THOUGHT 2: [your angle/approach]
[reasoning from this angle, using KG evidence]
Predicate: [predicate]

THOUGHT 3: [your angle/approach]
[reasoning from this angle, using KG evidence]
Predicate: [predicate]

THOUGHT 4: [your angle/approach]
[reasoning from this angle, using KG evidence]
Predicate: [predicate]

Begin:
```

**Phase 2 — Fusion pairwise (×2, pour chaque paire de nœuds) :**
```
{context}
--- KG EVIDENCE ---
{kg_evidence}

Two reasoning paths for "{place_name_subject} {relation_predicate} {place_name_object}":

Thought A:
{thought_a_content}
Predicate: {thought_a_predicate}

Thought B:
{thought_b_content}
Predicate: {thought_b_predicate}

Do these perspectives agree or disagree?
Synthesize them into a single, stronger conclusion.

Reasoning:
Answer: [
```

**Phase 3 — Agrégation finale :**
```
{context}
--- KG EVIDENCE ---
{kg_evidence}

Multiple merged reasoning paths for "{place_name_subject} {relation_predicate} {place_name_object}":

Path 1 (predicate: {merge_1_predicate}):
{merge_1_content}

Path 2 (predicate: {merge_2_predicate}):
{merge_2_content}

Considering ALL paths and the KG evidence, which single predicate
is the best final answer for "{relation_predicate}" between a {placetype_subject} ({geometry_type_subject}) and a {placetype_object} ({geometry_type_object})?

Final reasoning:
Answer: [
```

---

## 6. Résultats — Approche 1 : OSM Tool-Based (baseline)

Dans cette première approche, le KG est construit via des appels aux APIs **OpenStreetMap** (Nominatim, Overpass) pour récupérer les informations géographiques (hiérarchie administrative, lieux proches, métadonnées). Le LLM utilise ensuite ces données OSM comme contexte pour raisonner.

### 6.1. Accuracy globale (OSM)

| Stratégie | Accuracy | Correct / Total | Prédictions invalides |
|-----------|----------|-----------------|----------------------|
| **CoT** | **71.52%** | 231 / 323 | 2 |
| **ToT** | 63.16% | 204 / 323 | 43 |
| **GoT** | 58.51% | 189 / 323 | 70 |

### 6.2. Accuracy par prédicat (OSM)

| Prédicat | N | CoT | ToT | GoT |
|----------|---|-----|-----|-----|
| `within` | 80 | **98.8%** (79/80) | 93.8% (75/80) | 93.8% (75/80) |
| `contains` | 16 | **93.8%** (15/16) | **93.8%** (15/16) | **93.8%** (15/16) |
| `disjoint` | 79 | **88.6%** (70/79) | 78.5% (62/79) | 73.4% (58/79) |
| `crosses` | 17 | **58.8%** (10/17) | 23.5% (4/17) | 17.6% (3/17) |
| `touches` | 95 | **47.4%** (45/95) | 41.1% (39/95) | 31.6% (30/95) |
| `overlaps` | 36 | **33.3%** (12/36) | 25.0% (9/36) | 22.2% (8/36) |


### 6.4. Observations (OSM)

- **CoT est la meilleure stratégie** (71.52%) dans l'approche OSM. Contrairement aux attentes, ToT (63.16%) et GoT (58.51%) performent moins bien.
- **Taux élevé de prédictions invalides** : ToT produit 43 invalides (13.3%) et GoT 70 invalides (21.7%). Les prompts multi-branches/multi-nœuds combinés aux données OSM volumineuses provoquent des réponses mal formées.
- **`touches`** est massivement confondu avec `disjoint` (23 cas) — les données OSM de proximité ne capturent pas bien la notion de frontière partagée.
- **`overlaps`** est le plus faible (22–33%), souvent confondu avec `within` (13 cas).

---

## 7. Résultats — Approche 2 : Prompt-Based (KG sans LLM)

### 7.1. Accuracy globale (Prompt-Based)

| Stratégie | Accuracy | Correct / Total | Prédictions invalides |
|-----------|----------|-----------------|----------------------|
| **CoT** | 72.76% | 235 / 323 | 0 |
| **ToT** | **74.61%** | 241 / 323 | 0 |
| **GoT** | **74.61%** | 241 / 323 | 0 |

### 7.2. Accuracy par prédicat (Prompt-Based)

| Prédicat | N | CoT | ToT | GoT |
|----------|---|-----|-----|-----|
| `within` | 80 | **97.5%** (78/80) | 96.2% (77/80) | **97.5%** (78/80) |
| `contains` | 16 | **93.8%** (15/16) | **93.8%** (15/16) | **93.8%** (15/16) |
| `disjoint` | 79 | **84.8%** (67/79) | 83.5% (66/79) | 83.5% (66/79) |
| `crosses` | 17 | 52.9% (9/17) | 58.8% (10/17) | **64.7%** (11/17) |
| `touches` | 95 | 53.7% (51/95) | 58.9% (56/95) | **60.0%** (57/95) |
| `overlaps` | 36 | 41.7% (15/36) | **47.2%** (17/36) | 38.9% (14/36) |



## 8. Comparaison des deux approches

### 8.1. Accuracy globale comparative

| Stratégie | OSM Tool-Based | Prompt-Based | Δ (gain) |
|-----------|---------------|--------------|----------|
| **CoT** | 71.52% | 72.76% | **+1.24** |
| **ToT** | 63.16% | **74.61%** | **+11.45** |
| **GoT** | 58.51% | **74.61%** | **+16.10** |
| *Meilleur* | *71.52% (CoT)* | ***74.61% (ToT/GoT)*** | ***+3.09*** |


## 9. Analyse des résultats

### Pourquoi l'approche Prompt-Based surpasse l'approche OSM

1. **Zéro prédictions invalides** : L'approche prompt-based produit des prompts plus courts et plus ciblés (uniquement le triplet KG du CSV), ce qui permet au LLM de générer des réponses toujours bien structurées. L'approche OSM injectait des blocs de données OSM volumineux (hiérarchie administrative, lieux proches, métadonnées) qui « noyaient » le signal utile.

2. **Gain massif sur ToT et GoT** : Le gain le plus spectaculaire est sur ToT (+11.45) et GoT (+16.10). Dans l'approche OSM, les prompts multi-branches étaient trop longs (KG + branches/nœuds), provoquant des réponses mal formées et des prédictions invalides massives. L'approche prompt-based, avec ses prompts concis, permet à ces stratégies avancées de fonctionner correctement.

3. **Amélioration sur les prédicats difficiles** : `touches` (+12.6), `overlaps` (+13.9) et `crosses` (+5.9) sont nettement mieux capturés. Sans le bruit des données OSM, le LLM se concentre davantage sur l'analyse linguistique de l'expression vernaculaire et sur les contraintes géométriques.

4. **Léger recul sur `within` et `disjoint`** : L'approche OSM était légèrement meilleure sur `within` (98.8% vs 97.5%) et `disjoint` (88.6% vs 84.8%). Les données de hiérarchie administrative d'OSM aidaient à confirmer les relations d'inclusion et de séparation.

### Points forts globaux
- **`within`** et **`contains`** sont très bien reconnus dans les deux approches (93–99%).
- **`disjoint`** obtient 83–89% grâce à des marqueurs linguistiques clairs.
- L'approche prompt-based est **plus robuste** (0% invalide) et **plus rapide** (~3s/ligne vs ~30s/ligne pour OSM).

### Points faibles persistants
- **`touches`** (47–60%) reste le prédicat le plus confondu, principalement avec `within` et `disjoint`.
- **`overlaps`** (22–47%) est systématiquement le plus difficile, souvent confondu avec `within`.
- Ces confusions reflètent une ambiguïté intrinsèque du langage naturel : les expressions vernaculaires pour « toucher » et « chevaucher » sont souvent vagues.

### Comparaison des stratégies
- **Approche OSM** : CoT est la meilleure (71.52%). ToT et GoT souffrent du volume de données OSM.
- **Approche Prompt-Based** : ToT et GoT (74.61%) surpassent CoT (72.76%). Les stratégies multi-perspectives fonctionnent quand les prompts sont concis.
- **GoT** excelle sur `crosses` (64.7%) et `touches` (60.0%) grâce à sa structure en graphe.
- **ToT** est le meilleur sur `overlaps` (47.2%) grâce au mécanisme de vote.

---


## 10. Conclusion

L'approche prompt-based atteint **74.61% d'accuracy** (ToT/GoT), surpassant l'approche OSM tool-based (71.52% CoT) avec un gain de **+3.09 points** sur la meilleure stratégie de chaque approche, et jusqu'à **+16.10 points** sur GoT.

Le résultat le plus notable est l'**élimination complète des prédictions invalides** (0% vs 21.7% pour OSM GoT), permettant aux stratégies multi-perspectives (ToT, GoT) de réaliser leur plein potentiel. Les prédicats difficiles (`touches`, `overlaps`, `crosses`) bénéficient le plus de l'approche prompt-based, avec des gains de +5.9 à +13.9 points.

Les stratégies multi-perspectives (ToT, GoT) apportent un gain de **+1.85 points** par rapport au raisonnement linéaire (CoT) dans l'approche prompt-based, avec un avantage particulier de GoT sur les prédicats ambigus comme `crosses` et `touches`.
