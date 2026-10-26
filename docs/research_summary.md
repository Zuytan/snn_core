# Bilan de la piste « neurone natif CPU » (BPTN, LUT, forêt LUT)

25-26 octobre 2026, branche `research/lut-forest-shd`. Les mesures citées ont une ligne dans
[`CLAIMS.md`](../CLAIMS.md) ; les fichiers de résultats sont dans
[`tools/bptn_torch/results/`](../tools/bptn_torch/results/).

## 1. Question posée

L'idée de départ du projet BPTN : au lieu de reproduire le neurone biologique sur silicium, concevoir
un neurone à partir des forces d'un CPU de microcontrôleur (opérations sur des mots de bits, lectures
en table, branchements), et vérifier qu'il fait mieux que les modèles existants à coût égal sur une
vraie tâche, sur une vraie carte.

Critère fixé par le plan : au moins **2× moins de cycles à précision égale**, sur au moins **2 tâches
sur 3**.

## 2. Protocole commun

- **Carte** : Seeed Wio-E5 (STM32WLE5JC, Cortex-M4 sans FPU), 48 MHz et 2 états d'attente Flash
  (réglés et relus par le firmware), cycles mesurés au compteur DWT. Chaque modèle mesuré l'est sur
  tout le jeu de test, et chaque prédiction de la carte est comparée à une référence entière calculée
  sur l'hôte.
- **Entrée identique pour tous les modèles** : 64 canaux × 64 pas de temps, un bit par case.
- **Réglage à effort égal** : taux d'apprentissage et nombre d'époques choisis sur une validation
  (10 % de l'entraînement), puis 3 graines quand l'entraînement est aléatoire (SHD).
- **Tâches** :
  - **SHD** (Spiking Heidelberg Digits, 20 classes) : 700 canaux regroupés en 64, 1 s en 64 pas.
  - **Google Speech Commands v2** (35 mots) : 64 bandes mel × 64 trames, les 25 % de cases les plus
    énergétiques de chaque énoncé à 1.
  - **AutoHack** (intrusion CAN, 5 classes) : fenêtres de 64 trames, 64 bits par trame (identifiant,
    DLC, écart de temps court, 6 octets de données) ; fenêtres d'attaque et autant de fenêtres
    normales.

## 3. Chronologie des modèles

| Étape | Modèle | Idée | Résultat principal |
| :--- | :--- | :--- | :--- |
| 1 | **BPTN** | somme de popcounts de noyaux temporels de 64 bits | Équivalent exact d'un perceptron à poids ternaires ; sur SHD, 68,7 %, au niveau d'un réseau binarisé (69,7 %) avec 2× plus de Flash. Le Cortex-M4 n'a pas d'instruction POPCNT. |
| 2 | **Neurone LUT** | 6 bits de l'historique forment l'index d'une table apprise | Avec des connexions fixes, 54 % (2 000 LUT) ; avec des connexions apprises, 71,95 % sur la carte, mais à cycles égaux au même niveau que le réseau binarisé. |
| 3 | **Lire moins de modèle** | arrêt anticipé, calcul incrémental en streaming | −32 % et −38 % de cycles, sous le critère de 2×. |
| 4 | **Forêt LUT** (programmes de branchement appris) | arbres de décision dont chaque nœud est une LUT ; une inférence ne lit que quelques nœuds par arbre | Sur SHD, 74,82 % en 30 124 cycles mesurés : le seul modèle qui dépasse le critère. |
| 5 | **Références classiques** | random forest, gradient boosting, CNN int8 (CMSIS-NN) | La forêt LUT reste en tête sur SHD. |
| 6 | **Autres tâches** | Speech Commands et AutoHack | La forêt LUT perd, ou fait jeu égal sans avantage de coût. |
| 7 | **Forêt hybride** | comptages par canal avant la forêt | Rattrape le CNN sur le CAN, mais le gradient boosting fait autant avec 2× moins de cycles ; n'aide pas sur Speech Commands. |

## 4. Résultats

### SHD : mesures sur la carte (graine 0) et moyennes sur 3 graines (hôte)

| Modèle | Précision sur la carte | Moyenne hôte | Cycles par inférence | Flash |
| :--- | :---: | :---: | :---: | :---: |
| **Forêt LUT 64 × 4** | **74,82 %** | 74,31 % ± 0,93 (73,82 % ± 0,53 avec le réglage choisi sur validation) | **30 124** | 41,7 Ko |
| Forêt LUT 256 × 2 | 77,30 % | 75,40 % ± 1,77 | 78 914 | 37,8 Ko |
| Réseau binarisé, 100 neurones | 71,69 % | 71,72 % ± 0,27 | 216 758 | 55,1 Ko |
| Réseau binarisé, 64 neurones | 69,26 % | 69,73 % ± 0,57 | 142 163 | 35,9 Ko |
| Réseau LUT group-sum, 2 000 LUT | 71,95 % | 71,41 % ± 1,22 | 223 791 | 40,4 Ko |
| Gradient boosting 50 × 4 | 72,04 % | 73,14 % (taux 0,2) | 146 371 | 121,1 Ko |
| CNN int8 large (CMSIS-NN) | non mesuré | 74,37 % ± 0,62 | ≥ 562 000 (1,12 M MAC / 2) | 46,8 Ko |
| CNN int8 tiny | non mesuré | 66,62 % ± 1,18 | ≥ 17 700 (35 k MAC / 2) | 5,0 Ko |
| BPTN, 64 neurones | non mesuré | 68,73 % ± 0,31 | — | 67,2 Ko |
| LIF / LIF récurrent (snnTorch) | non mesuré | 60,6 % / 59,7 % | — | 6,0 / 10,1 Ko |

Les cycles des CNN sont une borne basse : `SMLAD` fait au plus 2 multiplications-accumulations par
cycle. À environ 74 %, le CNN demande donc au moins 18× les cycles de la forêt 64 × 4.

### Speech Commands et AutoHack (hôte, graine 0, précision en test)

| Modèle | Speech Commands | AutoHack | Coût indicatif |
| :--- | :---: | :---: | :--- |
| Forêt LUT 64 × 4 | 52,35 % | 70,09 % | ~30 k cycles (mesuré sur SHD) |
| Forêt LUT 256 × 2 | 56,13 % | 74,41 % | ~79 k cycles (mesuré sur SHD) |
| Forêt LUT 256 × 2, entrée hybride | 56,07 % | 78,12 % | idem, plus l'agrégation (non mesurée) |
| **CNN int8 tiny** | **61,32 %** | **77,75 %** | ≥ 17,8 k cycles, ~5 Ko |
| CNN int8 small | 71,80 % | 77,47 % | ≥ 67 k cycles |
| CNN int8 large | 79,56 % | 77,30 % | ≥ 562 k cycles |
| Gradient boosting 50 × 4 | 54,18 % | 76,41 % | 7 000 / 1 000 nœuds lus |
| Gradient boosting 50 × 4, entrée hybride | 47,93 % | 78,13 % | 1 000 nœuds sur AutoHack (~37 k cycles) |

**Critère de l'étape 2 non atteint** : la forêt LUT ne gagne que sur SHD.

## 5. Ce que l'on a appris

1. **Sur ce MCU, le coût suit les octets de modèle lus**, pas les opérations : environ 4 à 5,5 cycles
   par octet pour les modèles denses mesurés. Changer la forme du neurone (BPTN, LUT, binarisé) ne
   change rien tant que tout le modèle est lu à chaque inférence ; le calcul conditionnel (les arbres)
   est ce qui fait gagner.
2. **Un nœud LUT à entrées apprises vaut beaucoup de nœuds d'un bit** : sur SHD, la forêt LUT bat le
   gradient boosting de 2,8 points avec 4,9× moins de cycles et 2,9× moins de Flash, mesurés sur la
   carte.
3. **Mais elle ne sait pas compter.** Quand l'information est agrégée sur toute la fenêtre (CAN :
   comptages de trames ; parole : formes spectrales), les nœuds de 6 bits sous-apprennent. Ajouter
   des comptages en entrée corrige le CAN, mais profite autant au gradient boosting.
4. **Le CNN int8 reste une référence très forte** : très compact en Flash (5 Ko pour le plus petit),
   et c'est lui qui gagne sur Speech Commands.
5. **Mesurer sur silicium change les conclusions.** Plusieurs avantages annoncés en route ont été
   retirés après mesure : la carte tournait à 4 MHz et non à 48 MHz ; le compilateur précalculait
   certaines mesures ; un coût de micro-benchmark sur 5 Ko de modèle sous-estimait de 1,44× le coût
   sur 40 Ko ; l'avantage du réseau LUT sur le réseau binarisé a fondu de « 5 à 12 points » à zéro une
   fois les deux mesurés sur la carte. `CLAIMS.md` garde la trace de chaque retrait.

## 6. Conclusion et repositionnement

La forêt LUT n'est pas un modèle généraliste plus efficace pour microcontrôleur. C'est un modèle
efficace pour les **motifs temporels localisés dans des événements binaires**, comme les spikes de
SHD : sur cette tâche, il est le plus rapide de tout ce qui a été mesuré, à précision au moins égale.
Ailleurs, le CNN int8 et le gradient boosting font aussi bien ou mieux.

Le plan prévoyait ce cas : ne pas intégrer la forêt LUT au runtime comme modèle principal. Elle reste
un modèle expérimental, à présenter avec son domaine de validité. Le résultat (positif sur SHD,
négatif ailleurs, avec les mesures) reste publiable tel quel.

## 7. Questions ouvertes

- Cycles réels des CNN int8 sur la carte (firmware CMSIS-NN prêt, mesure interrompue).
- Consommation mesurée au PPK2 ou au Joulescope : toutes les énergies sont estimées.
- Tâches à événements natifs (caméra DVS, cochlée silicium), où la forêt LUT a le plus de chances de
  garder son avantage.
- Forêt dont les nœuds lisent aussi des comptages, ou mélange forêt et petite convolution.
- Une instruction de rassemblement de bits (de type `PEXT`) réduirait le coût d'un nœud LUT d'environ
  100 cycles à quelques cycles ; à évaluer sur RISC-V ou FPGA.
- Revue de littérature (WiSARD, DWN, LogicNets, machines de Tsetlin, arbres multivariés) avant toute
  revendication de nouveauté.

## 8. Reproduire

```bash
python3 -m venv ~/.cache/snn_core/venv
~/.cache/snn_core/venv/bin/pip install -r tools/bptn_torch/requirements-shd.txt
# SHD (téléchargement d'environ 169 Mo), forêt LUT 64 x 4 et export pour la carte :
~/.cache/snn_core/venv/bin/python tools/bptn_torch/train_shd_comparison.py --download --models bp \
    --bp-trees 64 --bp-depth 4 --export-bp ~/.cache/snn_core/bp_export_t64_d4
# Exécution sur la carte (sonde ST-Link, hors bac à sable) :
python3 tools/run_lut_shd_on_board.py --kind bp --chunk 256 \
    --model-dir ~/.cache/snn_core/bp_export_t64_d4 --out bp_board.json
```

Les autres modèles et tâches utilisent les mêmes scripts : `--models bnn|lut|lif|rlif`,
`tree_baselines.py`, `cnn_int8_baseline.py`, `encode_speech_commands.py`, `encode_autohack.py`,
`make_hybrid_features.py` (voir [`tools/bptn_torch/README.md`](../tools/bptn_torch/README.md)).
