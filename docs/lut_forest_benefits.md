# Forêt d'arbres LUT : bénéfices réels et perspectives

26 octobre 2026. Bilan des expériences de la branche `research/lut-forest-shd`. Les mesures de ce
document ont une ligne dans [`CLAIMS.md`](../CLAIMS.md) ; les plages de cycles viennent des fichiers
`*_board_*.json` de [`tools/bptn_torch/results/`](../tools/bptn_torch/results/), et les valeurs
calculées (octets lus, énergie, taux d'occupation) sont signalées comme telles.

## Le modèle en bref

Une forêt de T arbres de décision binaires de profondeur D. Chaque nœud est un **neurone LUT** : il
lit 6 bits à des positions apprises de l'historique d'entrée, forme un index de 6 bits et lit sa
sortie (0 ou 1) dans une table apprise de 64 entrées ; cette sortie choisit l'enfant gauche ou droit.
La feuille atteinte ajoute 20 scores int8, un par classe ; la classe prédite est celle du score
maximal. Une inférence lit donc T × D nœuds et T feuilles, pas tout le modèle.

L'entraînement (`tools/bptn_torch/train_shd_comparison.py --models bp`) route de façon stricte en
propagation avant, comme sur la carte, et fait passer le gradient par les probabilités de routage
(estimateur straight-through). Les positions lues par chaque nœud sont apprises comme dans les
Differentiable Weightless Neural Networks. Le firmware est
[`bp_shd_eval.rs`](../examples/wio_e5_benchmark/src/bin/bp_shd_eval.rs).

Protocole commun à tous les chiffres ci-dessous : Spiking Heidelberg Digits (20 classes), 700 canaux
regroupés en 64, la première seconde découpée en 64 pas, un bit par case ; même répartition
entraînement / validation / test ; jeu de test complet (2 264 échantillons) exécuté sur un Seeed
Wio-E5 (STM32WLE5JC, Cortex-M4 sans FPU, 48 MHz, 2 états d'attente Flash), cycles mesurés par le
compteur DWT.

## 1. Bénéfices mesurés

| Modèle (mesuré sur la carte) | Précision (graine 0) | Moyenne sur 3 graines (hôte) | Cycles par inférence | Temps | Flash du modèle |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Forêt LUT 64 × 4** | **74,82 %** | 74,31 % ± 0,93 | **30 124** | **0,63 ms** | **41,7 Ko** |
| Forêt LUT 256 × 2 | 77,30 % | 75,40 % ± 1,77 | 78 914 | 1,64 ms | 37,8 Ko |
| Forêt LUT 128 × 4 | 75,22 % | 74,66 % ± 0,49 | 102 734 | 2,14 ms | 81,4 Ko |
| Réseau binarisé, 100 neurones | 71,69 % | 71,72 % ± 0,27 | 216 758 | 4,52 ms | 55,1 Ko |
| Gradient boosting 50 itérations × profondeur 4 | 72,04 % | — | 146 371 | 3,05 ms | 121,1 Ko |
| Réseau LUT group-sum, 2 000 LUT | 71,95 % | 71,41 % ± 1,22 | 223 791 | 4,66 ms | 40,4 Ko |

- **Efficacité** : la forêt 64 × 4 fait +2,6 points avec 7,2× moins de cycles et 24 % de Flash en
  moins que le meilleur réseau binarisé mesuré, et +2,8 points avec 4,9× moins de cycles et 2,9× moins
  de Flash que le meilleur gradient boosting qui tient sur la carte. C'est le seul modèle de la série
  qui dépasse le critère de 2× fixé par le plan.
- **Le gain vient des nœuds LUT, pas seulement des arbres** : sur la même entrée, le gradient
  boosting à nœuds d'un bit n'atteint 75 % qu'avec 7 997 nœuds lus et 236,8 Ko (hôte), et le random
  forest demande 11,4 Mo pour 71,6 %. À Flash comparable (23,5 Ko), le gradient boosting fait 60,5 %.
- **Latence prévisible** : de 30 102 à 30 147 cycles sur les 2 264 échantillons (±0,1 %), contre
  ±1,3 % pour le réseau binarisé et de 143 640 à 160 941 cycles pour le gradient boosting.
- **Déploiement exact** : les 2 264 prédictions de la carte sont identiques à la référence entière
  calculée sur l'hôte.
- **Matériel minimal** : décalages, ET logiques et lectures mémoire ; aucune multiplication, aucune
  FPU, aucune instruction DSP.

## 2. Bénéfices par construction (vrais, pas encore exploités)

- **Coût indépendant de la taille de l'entrée.** Une inférence lit 64 × 4 nœuds quelle que soit la
  largeur de l'historique (4 096 bits ici). Élargir l'entrée (plus de canaux, plus de temps) coûte à
  l'entraînement, pas à l'inférence.
- **Peu de modèle lu par inférence** : 6 400 octets (256 nœuds et 64 feuilles de 20 octets) sur
  41,7 Ko. Sur ce MCU, le coût suit les octets de modèle lus (4 à 5,5 cycles par octet pour les
  modèles denses mesurés) ; la forêt tire parti de cette loi au lieu de la subir.
- **Inférence « à tout moment »** : les arbres sont indépendants ; on peut s'arrêter après N arbres
  quand le budget de temps est atteint, avec une précision qui baisse progressivement (non mesuré).
- **Pire cas calculable à la main** : chaque arbre suit un chemin de longueur fixe D, sans boucle
  dépendant des données. C'est un atout pour les systèmes temps réel et de sûreté.
- **Adaptation sur l'appareil** : les feuilles sont une table de scores linéaire en sortie ; on peut
  en principe les réajuster sur le MCU avec quelques exemples, sans rétropropagation à travers les
  nœuds (non testé).
- **Portabilité** : rien de propre au Cortex-M4 ; le même code convient au Cortex-M0+ et au RISC-V
  (non mesuré sur ces cibles).

## 3. Hypothèses et limites

- **Le gain ne se généralise pas (étape 2, 26 octobre 2026).** Sur Google Speech Commands (35 mots) et
  sur le jeu CAN AutoHack, avec le même encodage 64 × 64 et le même protocole, le plus petit CNN int8
  bat toutes les forêts essayées : 61,3 % contre 56,1 % et 77,8 % contre 74,4 %, pour un coût en
  cycles plus faible et 8 à 10× moins de Flash. Les forêts sous-apprennent sur ces tâches, où
  l'information est agrégée sur toute la fenêtre. Les bénéfices de ce document valent donc pour SHD,
  pas encore au-delà.
- **Référence int8 CMSIS-NN : partielle.** Un CNN int8 réglé de la même façon atteint 74,4 % avec
  1,12 M multiplications-accumulations, donc au moins 562 k cycles même au débit maximal de `SMLAD`
  (au moins 18× la forêt). Il est en revanche plus compact en Flash à précision plus basse (5 Ko pour
  66,6 %). Ses cycles sur la carte restent à mesurer.
- **Loin de l'état de l'art en précision** : les meilleurs modèles publiés sur SHD dépassent
  largement 75 %, à un coût bien plus élevé. Le résultat porte sur l'efficacité, pas sur la
  précision maximale.
- **Pas de streaming** : les positions apprises sont absolues dans la fenêtre ; le modèle n'est pas
  invariant au décalage et suppose une fenêtre alignée.
- **Variance d'entraînement** de ±1 à 1,8 point selon la graine. L'entraînement est lourd : un score
  par position d'entrée pour chaque entrée de nœud (98 M de scores pour 4 000 nœuds).
- **Énergie non mesurée** : environ 8,7 µJ par inférence en estimation (0,63 ms × 13,86 mW, courant
  supposé de 4,2 mA). Il faut une mesure au PPK2 ou au Joulescope.
- **Nouveauté à vérifier** : l'idée est proche des réseaux sans poids (WiSARD, DWN), de LogicNets et
  des arbres à coupes multivariées. Une revue de littérature est nécessaire avant de la présenter
  comme nouvelle.
- **Gradient boosting peu réglé** : seuls la profondeur et le nombre d'itérations ont été explorés.
  En multiclasse, il construit un arbre par classe à chaque itération, ce qui le pénalise sur
  20 classes.

## 4. Ce que cela permet

### Court terme (semaines) : consolider et rendre utilisable

- Valider sur deux ou trois autres tâches, dont une tâche CAN (fenêtre courte, latence garantie) et
  Google Speech Commands ; ajouter la référence int8 CMSIS-NN et une mesure de courant.
- Faire de la forêt un type d'inférence du runtime (`LutForest` : `no_std`, modèle en Flash, sans
  tas), avec son exportateur dans `snn-compile`. C'est le premier modèle du dépôt qui justifie une
  API stable.
- Cibles d'application : détecteurs toujours actifs (réveil sur événement, anomalie vibratoire,
  intrusion CAN) sur des MCU bon marché sans DSP. À 0,63 ms par décision et une inférence par
  seconde, le CPU est occupé 0,063 % du temps (capteur et radio non compris).

### Moyen terme (mois) : exploiter les propriétés structurelles

- **Streaming** : positions relatives au pas courant et mise à jour incrémentale ; un arbre ne lit
  que 4 positions, donc seuls les arbres touchés par les bits qui changent sont réévalués.
- **Entrées multi-échelles** (pyramide temporelle de type H-BPTN) pour couvrir plusieurs secondes
  sans augmenter le coût d'inférence.
- **Cascades** : une petite forêt d'environ 10 k cycles filtre, un modèle plus coûteux ne tourne que
  sur les cas ambigus.
- **Personnalisation sur l'appareil** en réajustant seulement les feuilles.
- **Publication** : article court (par exemple tinyML Research Symposium) avec le protocole
  reproductible de ce dépôt (scripts, carte, `CLAIMS.md`).

### Long terme (années) : une primitive pensée pour le silicium

- Un nœud LUT est la primitive native du FPGA (une LUT6) et un bloc très simple en ASIC. Sur CPU, il
  ne lui manque qu'une instruction : rassembler 6 bits épars en un index (comme `PEXT` sur x86, ou
  une instruction personnalisée RISC-V). Elle ramènerait un nœud d'environ 100 cycles à quelques
  cycles : le modèle définit l'instruction, et non l'inverse.
- Plus largement : le calcul vu comme des lectures mémoire à adressage appris, avec un calcul
  conditionnel. Ce paradigme ne copie ni la biologie ni l'algèbre dense des GPU, ce qui était
  l'objectif initial du projet BPTN.
- Risque : les NPU intégrés aux nouveaux MCU (par exemple Ethos-U55). Le créneau de la forêt LUT
  serait alors les MCU sans NPU, les garanties de pire cas, et le matériel minimal.

## En une phrase

Aujourd'hui, la forêt LUT est un classifieur de fenêtres binaires 5 à 7× plus efficace en cycles que
les alternatives mesurées sur la même carte. Sa promesse à long terme est d'être la primitive d'un
neurone pensé pour le silicium, à condition de la valider sur d'autres tâches et face à CMSIS-NN.
