# Plan de consolidation — forêts et automates LUT

Date : 28 octobre 2026. Branche : `research/lut-forest-shd`. Fait suite au bilan
[`research_summary.md`](research_summary.md) et au [`plan_de_resolution.md`](plan_de_resolution.md).

**Pourquoi ce plan.** Depuis l'étape 2, chaque variante (entrée hybride, banque multi-échelles,
automate, boosting, distillation) a rapporté 0 à 4 points, presque toujours sur une seule graine,
alors que l'écart d'une graine à l'autre est du même ordre (± 2,9 points pour l'automate sur
AutoHack, environ 4 points pour la forêt sur MIT-BIH). Par ailleurs, les cycles du CNN int8, notre
principal concurrent, ne sont qu'une borne basse (MAC / 2). On arrête donc d'empiler des variantes :
on établit le vrai classement (WP1), on mesure les concurrents sur la carte (WP2), on présente le
tout en courbes précision / cycles (WP3) et on prépare l'article (WP4). La recherche ciblée (WP5)
ne vient qu'ensuite.

## 0. Règles communes

- **Protocole figé.** On garde, pour chaque tâche, les réglages déjà retenus pour la graine 0 (ils
  diffèrent d'une tâche à l'autre : par exemple `--lr-lut 1e-2` pour la forêt 64 × 4 de SHD,
  `5e-3` ailleurs) ; WP1 ne relance aucun réglage. Sur AutoHack et MIT-BIH, la validation n'est pas disjointe par enregistrement ou par
  patient et dépasse le test de 10 à 30 points : elle ne sert qu'à choisir l'époque. On ne rapporte
  que la précision en test du modèle déployé (`test_acc_deployed` pour la forêt, référence entière
  pour le boosting et l'automate, int8 pour le CNN).
- **Graines appariées.** Chaque `--seed` change le découpage de validation (10 %) et
  l'initialisation ; toutes les familles d'une même tâche utilisent les mêmes graines. Le boosting
  LUT (sans sous-échantillonnage) et le gradient boosting sont déterministes à découpage fixé : leur
  écart-type ne mesure que l'effet du découpage.
- **Règle statistique** (implémentée par `aggregate_seeds.py`, WP1.0). Pour deux modèles A et B
  d'une même tâche, on calcule la différence appariée d = A − B sur les graines communes.
  « A > B » si moyenne(d) > 2 × écart-type(d) **et** d > 0 sur toutes les graines ; sinon
  « égalité ». n = 3 graines, n = 5 sur MIT-BIH (dispersion plus forte). Les cycles ne sont pas
  moyennés : ils dépendent de l'architecture, pas de la graine.
- **Exécution.** Depuis `tools/bptn_torch/`, avec le venv `~/.cache/snn_core/venv`, sous
  `caffeinate -i` et sur secteur (le Mac s'est déjà mis en veille sur batterie pendant 4 h). Les
  runs vont dans `results/seeds/<tâche>/<modèle>_seed<s>.json`. Les JSON existants de la graine 0
  sont **copiés** (pas déplacés) vers ces noms, pour ne pas casser les liens de la documentation.
- **Interdit.** Aucune nouvelle variante sur une seule graine ; aucun réglage sur une validation qui
  fuit ; aucun chiffre « mesuré » qui ne sort pas de la carte.

Variables à définir avant les commandes (MIT-BIH : intervalles RR causaux) :

```bash
cd tools/bptn_torch && PATH=$HOME/.cache/snn_core/venv/bin:$PATH
C=$HOME/.cache/snn_core
export SHD=$C/shd/shd_encoded_c64_t64_1.00s.npz AH=$C/autohack/autohack_w64.npz \
  AHMS=$C/autohack/autohack_multiscale.npz GSC=$C/gsc/gsc_encoded_b64_f64.npz MIT=$C/mitdb/mitbih_w64_rrc.npz
```

## WP1. Consolidation sur 3 graines, sans la carte (priorité 1)

### WP1.0 Script d'agrégation `aggregate_seeds.py` (à écrire)

- **Entrées** : `--glob "results/seeds/autohack/*.json"`, `--metric auto` (un adaptateur par
  script : clé `test_acc_deployed` pour la forêt, `test_acc_int8` pour le boosting,
  `integer_test_acc` pour l'automate, `test_int` pour le CNN, clé de test du gradient boosting à
  relire dans son JSON), `--select CONFIG` pour les fichiers qui contiennent plusieurs
  configurations (`cnn_int8_baseline.py`, `tree_baselines.py`), `--ref MODÈLE`,
  `--out results/summary/<tâche>.json`, `--md` (tableau Markdown).
- **Sortie** : par modèle, n, moyenne, écart-type (ddof = 1), valeurs par graine, cycles et Flash
  avec leur statut (`measured` ou `estimated`) ; par paire (modèle, référence), moyenne(d),
  écart-type(d) et verdict selon la règle du §0.
- **Cas limites** : graines manquantes (comparer sur les graines communes et afficher n) ; anciens
  fichiers de la graine 0 sans suffixe.
- **Critère de sortie** : retrouver exactement 74,31 ± 0,93 sur
  `results/bp_sweep/bp_t64_d4_seed{0,1,2}.json`.
- **Temps** : environ 2 h de développement.

### WP1.1 AutoHack (graines 1 et 2 ; l'automate a déjà ses 3 graines)

```bash
for s in 1 2; do
  python -u train_shd_comparison.py --data-npz $AH --models bp --bp-trees 256 --bp-depth 2 --lr-lut 5e-3 --epochs 40 --seed $s --out results/seeds/autohack/bp256x2_seed$s.json
  python -u train_lut_boosting.py --data-npz $AH --trees 256 --depth 2 --lr 0.1 --lam 10 --seed $s --out results/seeds/autohack/boost256x2_seed$s.json
  python -u tree_baselines.py --data $AH --rf "" --hgb 50x4 --hgb-lr 0.1 --seed $s --out results/seeds/autohack/hgb_seed$s.json
  python -u tree_baselines.py --data $AH --rf "" --hgb 50x4 --hgb-lr 0.2 --seed $s --out results/seeds/autohack/hgb02_seed$s.json
  python -u tree_baselines.py --data $AHMS --rf "" --hgb 50x4 --hgb-lr 0.1 --seed $s --out results/seeds/autohack/hgb_ms_seed$s.json
  python -u cnn_int8_baseline.py --data $AH --configs tiny,small --lr 1e-3 --epochs 40 --seed $s --out results/seeds/autohack/cnn_seed$s.json
done
```

- `--rf ""` désactive bien la forêt aléatoire (`tree_baselines.py` ignore une liste vide).
- `--lr 1e-3` : réglage du CNN tiny à 77,75 % (`results/step2_autohack/cnn_lr1e-3_seed0.json`).
- Gradient boosting sur l'entrée brute : le chiffre de référence (76,41 %) vient du taux 0,2,
  retenu sur la validation (`results/step2_autohack/hgb_lr0.2.json`) ; le taux 0,1 donne 74,16 %.
  On garde les deux, le taux 0,1 servant de comparaison directe avec la banque multi-échelles.
- La banque multi-échelles fait environ 2 Go en mémoire : lancer ces runs seuls.
- **Temps** : environ 4 h (à confirmer au premier run).

### WP1.2 Speech Commands (graines 1 et 2)

Un professeur par graine : celui de la graine 0 a vu une partie de la validation des graines 1 et 2.

```bash
for s in 1 2; do
  python -u cnn_int8_baseline.py --data $GSC --configs large --lr 3e-3 --epochs 40 --seed $s --save-teacher $C/gsc/teacher_cnn_large_seed$s.npy --out results/seeds/gsc/teacher_seed$s.json
  python -u train_shd_comparison.py --data-npz $GSC --models bp --bp-trees 256 --bp-depth 2 --lr-lut 5e-3 --epochs 40 --teacher $C/gsc/teacher_cnn_large_seed$s.npy --distill-alpha 0.7 --distill-temp 4 --time-shift 8 --seed $s --out results/seeds/gsc/bp256x2_both_seed$s.json
  python -u cnn_int8_baseline.py --data $GSC --configs tiny,small --lr 3e-3 --epochs 40 --seed $s --out results/seeds/gsc/cnn_seed$s.json
done
```

- `--lr 3e-3` : réglage des CNN tiny (61,32 %) et large (79,56 %) sur Speech Commands.
- **Temps** : environ 3 h.

### WP1.3 MIT-BIH avec RR causal (graines 1 à 4)

- Forêt 64 × 4 (`--bp-trees 64 --bp-depth 4 --lr-lut 5e-3 --epochs 40`), gradient boosting
  (`tree_baselines.py --rf "" --hgb 50x4 --hgb-lr 0.1`), CNN tiny (`--configs tiny --lr 2e-3`,
  valeur par défaut du script : la commande d'origine de
  `results/step2_mitbih_rr_causal/cnn_tiny_seed0.json` n'enregistre pas le taux ; relancer aussi la
  graine 0 si l'on n'est pas sûr qu'elle a utilisé ce défaut).
- On garde le `.npz` actuel : la graine d'`encode_mitbih.py` ne change que le plafonnement des
  classes, pas le découpage DS1 / DS2 entre patients.
- Rappel par classe avec `mitbih_recall.py` pour chaque graine : c'est la métrique attendue en ECG.
- **Temps** : environ 2 h.

### WP1.4 SHD, boosting LUT 256 × 2 (graines 1 et 2)

```bash
for s in 1 2; do
  python -u train_lut_boosting.py --data-npz $SHD --trees 256 --depth 2 --lr 0.1 --lam 10 --seed $s --out results/seeds/shd/boost256x2_seed$s.json
done
```

- **Temps** : environ 15 min.

### Sorties et décision de WP1

- **Sorties** : `results/summary/{shd,autohack,gsc,mitbih}.json` et un tableau moyenne ± écart-type
  par tâche, reporté dans `research_summary.md` et `CLAIMS.md` (chaque ligne passe de « 1 seed » à
  « n = 3 » ou « n = 5 »).
- **Décision** : pour chaque tâche, la forêt, le boosting et l'automate « gagnent », font
  « égalité » ou « perdent » contre le CNN tiny et le gradient boosting, selon la règle du §0.
  - Si le boosting LUT 256 × 2 sur SHD reste ≥ 74,3 % avec d > 0 contre la forêt 64 × 4 sur les
    3 graines, il devient un candidat SHD de WP2.
  - Une tâche où l'on n'obtient qu'une égalité passe en « parité, à coût à mesurer » et sort des
    victoires.
- **Dépendances** : aucune. **Calcul total** : environ 10 h, soit 2 nuits.

## WP2. Mesures sur la carte (dès que le Wio-E5 est rebranché)

`probe-rs` doit tourner hors du bac à sable. Le CNN demande `CMSIS_NN_DIR`, `CMSIS_CORE_DIR` (et
`CMSIS_AR` si besoin) ; voir `examples/wio_e5_benchmark/build.rs`.

### WP2.1 CNN int8 sur SHD (la mesure qui peut changer le verdict de l'étape 2)

```bash
# Réglages retenus sur la validation de la graine 0 (results/tuning/cnn_<config>_lr*_e*.json) :
# tiny lr 3e-3, 80 époques ; small lr 3e-3, 40 époques ; large lr 1e-3, 40 époques.
for spec in "tiny 3e-3 80" "small 3e-3 40" "large 1e-3 40"; do
  set -- ${=spec}
  python -u cnn_int8_baseline.py --data $SHD --configs $1 --lr $2 --epochs $3 --seed 0 --export-dir $C/cnn_export_seed0 --out results/cnn_export_seed0_$1.json
  python ../run_lut_shd_on_board.py --kind cnn --model-dir $C/cnn_export_seed0/cnn_export_$1 --chunk 200 \
    --env CMSIS_NN_DIR=$C/cmsis/cmsis-nn --env CMSIS_CORE_DIR=$C/cmsis/cmsis6 --out results/cnn_board_$1.json
done
```

- **Contrôle** : prédictions identiques à 100 % à la référence entière de l'hôte.
- **Règle de décision** : r = cycles mesurés / borne (MAC / 2).
  - r ≥ 2 : on refait le verdict de l'étape 2 sur toutes les tâches, avec les cycles CNN
    extrapolés par r jusqu'à leur mesure en WP2.4.
  - r < 1 : bug dans la borne ou dans le comptage, à chercher avant tout.
- **Temps** : environ 1 h de carte.

### WP2.2 Boosting LUT sur SHD

```bash
python -u train_lut_boosting.py --data-npz $SHD --trees 256 --depth 2 --lr 0.1 --lam 10 --export $C/lutboost_export_t256d2 --out results/lutboost_shd/t256d2_export.json
python ../run_lut_shd_on_board.py --kind bp --model-dir $C/lutboost_export_t256d2 --chunk 256 --out results/lutboost_board_t256d2.json
```

- **Critère** : identité bit à bit avec la référence entière (75,66 % en test pour la graine 0), et
  cycles proches de `results/bp_board_t256_d2.json` (même format, donc même coût attendu).

### WP2.3 Généraliser le firmware aux autres tâches (à écrire)

- `build.rs` : dans les sections BP, TREE et CNN, lire `classes` (et pour le CNN `channels`,
  `ticks`) dans `meta.json` et émettre des constantes ; garder 20 / 64 / 64 si la clé manque
  (compatibilité SHD).
- Retirer les constantes en dur de `bp_shd_eval.rs` (`CLASSES`, l. 27), `cnn_shd_eval.rs`
  (`CLASSES`, `CHANNELS`, `TICKS`, l. 31-33) et `tree_shd_eval.rs` le cas échéant.
- Exporteurs (`train_shd_comparison.py --export-bp`, `train_lut_boosting.py --export`,
  `tree_baselines.py --export`, `cnn_int8_baseline.py --export-dir`) : écrire `classes`,
  `channels` et `ticks` dans `meta.json`.
- **Cas limites** : CLASSES ≥ 2 pour l'argmax ; Flash de 256 Ko partagée entre modèle et lot de
  test (`--chunk` par tâche) ; dimensions CMSIS-NN en `i32`.
- **Critère** : les résultats SHD restent identiques bit à bit après le changement ; commandes de
  validation d'`AGENTS.md` passées. Vérifiable par compilation sans la carte.

### WP2.4 Mesures hors SHD

- Le coût ne dépend pas des données (profondeur fixe, CNN dense) : 1 ou 2 lots par modèle suffisent,
  avec contrôle d'identité ; la précision reste celle de WP1.
- AutoHack : forêt 256 × 2, boosting 256 × 2, gradient boosting 50 × 4, CNN tiny. Speech Commands :
  forêt 256 × 2 distillée, CNN tiny. MIT-BIH : forêt 64 × 4, CNN tiny.
- **Dépend de** WP2.3. **Temps** : environ 2 h de carte.

### WP2.5 Firmware de l'automate LUT (à écrire)

- **Export** : `train_lut_automaton.py --export DIR` (à écrire) écrit `meta.json` (M, S, k,
  mots par trame = 2, classes, trames), les positions de bits (`u8`, 0 à 63), la table des états
  suivants M × S × 2^k empaquetée sur 4 bits (8 Ko pour 16 × 16), les scores int8 [M][S][C] et
  `host_pred.bin` produit par la **référence entière** existante.
- **Firmware** : `src/bin/automaton_eval.rs`, une section `AUT_MODEL_DIR` dans `build.rs`,
  `--kind automaton` dans `run_lut_shd_on_board.py`. Boucle par trame : pour chaque automate,
  rassembler 6 bits depuis les 2 mots de la trame, lire le quartet de la table, mettre à jour l'état,
  ajouter les scores int8 dans des accumulateurs `i32`.
- **Cas limites** : quartet haut ou bas selon la parité de l'adresse ; S > 16 impose des octets ;
  état initial 0 ; même règle d'égalité dans l'argmax que l'hôte.
- **Critères** : identité 100 % sur 2 lots AutoHack ; cycles par trame ≤ 1,5 × l'estimation
  (2,2 k). Sinon, on documente l'écart et l'estimation est retirée.
- **Dépend de** WP2.3 (même patron de `build.rs`).

## WP3. Courbes précision / cycles

- **Script** `plot_pareto.py` (à écrire) : entrées `results/summary/*.json` (WP1) et
  `results/*board*.json` (WP2) ; sorties `docs/figures/pareto_<tâche>.png` et un CSV.
- **Contenu** : cycles par inférence en échelle logarithmique, précision moyenne ± écart-type ;
  **marqueurs pleins = cycles mesurés, creux = estimés** (borne MAC / 2, estimation de l'automate) ;
  un panneau par tâche, plus un panneau Flash ; front de Pareto calculé sur les seuls points mesurés.
  Un point à une seule graine est grisé et annoté « 1 seed ».
- **Critère** : chaque point est relié à son fichier source dans le CSV.
- **Dépend de** WP1 et au moins WP2.1. **Temps** : environ 3 h.

## WP4. Revue de la littérature et article (en parallèle de WP1 et WP2)

### Revue

Pour chaque famille : méthode d'apprentissage, format d'inférence, cible matérielle, mesure sur MCU
ou non, résultats sur SHD, Speech Commands ou ECG.

- WiSARD et réseaux RAM ; **DWN** (Differentiable Weightless Neural Networks, connexions des LUT
  apprises : le plus proche de notre apprentissage) ;
- LogicNets et PolyLUT (FPGA) ; difflogic et sa version convolutive ;
- machines de Tsetlin, y compris convolutives, et leurs implémentations sur MCU ;
- arbres pour MCU : **Bonsai** et ProtoNN (EdgeML), antériorité à examiner en premier ;
- arbres souples ou différentiables (Deep Neural Decision Forests, TAO), arbres obliques ou
  multivariés, gcForest ;
- automates finis et automates pondérés appris.

Les références exactes sont à retrouver et à vérifier ; aucune n'est citée ici de mémoire comme
établie.

### Énoncé de nouveauté à vérifier

Une forêt de programmes de branchement dont chaque nœud est une LUT de 6 bits à entrées apprises
(par estimateur straight-through ou par boosting glouton), sans multiplication, mesurée en cycles
réels sur Cortex-M4 et comparée à CMSIS-NN int8 et au gradient boosting avec un protocole à
plusieurs graines.

- **On garde** cet énoncé si aucun travail (DWN, Bonsai et les machines de Tsetlin sur MCU en
  premier) ne couvre la combinaison « nœuds LUT appris + arbres + mesure sur MCU ».
- **Sinon**, on recentre l'article sur l'étude comparative mesurée.

### Article

- **Cibles** : tinyML Research Symposium en priorité, sinon ICONS ou NICE (dates limites à
  vérifier).
- **Plan** : introduction ; modèle (nœud LUT, arbre, forêt, automate) ; apprentissage (STE,
  boosting, distillation) ; inférence sur MCU (format, boucle, Flash) ; protocole (encodage
  64 × 64 bits, graines, découpages, limites de la validation) ; résultats sur 4 tâches ; limites
  (une seule carte, un seul encodage, MIT-BIH bruité).
- **Figures** : schéma du nœud LUT et du parcours ; précision / cycles mesurés sur SHD ; grille des
  4 tâches ; tableau moyenne ± écart-type et verdicts ; répartition des cycles d'un nœud ; précision
  de l'automate selon le nombre de trames lues.
- **Temps** : environ 4 jours de rédaction.

## WP5. Recherche ciblée (seulement après WP1 et WP2)

### WP5.1 Automate avec une perte à chaque trame

- `train_lut_automaton.py --per-frame-loss {none,mean,linear}` (à écrire) : entropie croisée sur
  les scores cumulés à chaque trame, avec un poids uniforme ou croissant dans le temps.
- AutoHack, 16 automates × 16 états, 20 époques, 3 graines, dans `results/automaton_autohack_pf/`.
- **Critère** : précision après 8 trames ≥ 65 % (54 % aujourd'hui), sans perdre plus d'un point en
  fin de fenêtre selon la règle du §0. **Temps** : environ 6 h.

### WP5.2 Tâche à événements natifs : DVS Gesture

- Données via la bibliothèque `tonic` (taille du téléchargement à vérifier) ; découpage officiel
  disjoint par utilisateur ; 11 classes.
- `encode_dvsgesture.py` (à écrire) : réduction spatiale 128 × 128 → 8 × 8 (polarités fusionnées,
  64 canaux), 64 intervalles de temps, un bit par case selon un seuil choisi sur l'entraînement ;
  sortie `~/.cache/snn_core/dvsgesture/dvsg_c64_t64.npz` au même format que les autres tâches.
- Forêt 64 × 4 et 256 × 2, boosting LUT, CNN tiny, 3 graines.
- **Critère** : forêt ≥ CNN tiny − 1 point avec au plus la moitié de ses cycles mesurés.
  **Temps** : environ 1 jour, téléchargement compris.

## Ménage

- Décider du sort de `tools/bptn_torch/compress_ecg_bench.py` (non suivi) : le commiter ou le
  supprimer.
- Pousser `research/lut-forest-shd` ; l'ouverture d'une PR vers `main` reste une décision de
  l'utilisateur.
- Lancer les commandes de validation d'`AGENTS.md` avant tout commit qui touche le firmware.

## Calendrier

| Semaine | Contenu |
| :--- | :--- |
| S1 | WP1.0 ; nuits 1 et 2 : WP1.1 à WP1.4 ; début de WP4 |
| S2 | Tableaux et verdicts de WP1 ; WP2.3 (vérifiable par compilation) ; revue WP4 |
| S3 (carte branchée) | WP2.1, WP2.2, WP2.4 ; décision sur r |
| S4 | WP2.5 (automate) ; WP3 (figures) |
| S5–S6 | Rédaction de l'article ; WP5.1 puis WP5.2 si le temps le permet |

## Risques

| Risque | Effet | Parade |
| :--- | :--- | :--- |
| Les 3 graines effacent l'avance du boosting ou de la forêt | Verdicts en « égalité » | Publier la parité à coût mesuré ; ne pas re-régler |
| CNN réel ≫ borne MAC / 2 | Le verdict de l'étape 2 change en faveur des LUT | Règle r ≥ 2, verdict refait |
| CNN réel ≈ borne | La forêt perd hors SHD | Recentrer l'article sur SHD et l'étude comparative |
| Flash insuffisante (modèle + lot) hors SHD | Mesure impossible | Lots réduits ; un lot suffit pour les cycles |
| `probe-rs` bloqué par le bac à sable | Pas de mesure | Lancer hors bac à sable |
| Antériorité (DWN, Bonsai, Tsetlin sur MCU) | Nouveauté réduite | Recentrer l'énoncé sur les mesures MCU et le protocole |
| Validation qui fuit (AutoHack, MIT-BIH) | Choix biaisés | Réglages figés ; rapporter le test seul |
| Mémoire (banque multi-échelles 2 Go) ou mise en veille | Runs perdus | Runs seuls, `caffeinate -i`, sur secteur |
| Automate en firmware ≠ référence entière | Pas de mesure | Comparer trame par trame sur un échantillon avant le lot complet |
