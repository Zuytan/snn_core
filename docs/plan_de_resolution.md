# snn_core — Plan de résolution

Oct 25, 2026 · @Alexandre

## Objectif et critères de succès

En 9 mois, faire de snn_core un runtime SNN pour MCU qu'un industriel peut évaluer sans trouver une seule affirmation fausse, utiliser légalement, et dont les gains se reproduisent sur une tâche réelle. Le plan suit l'ordre des problèmes relevés dans l'analyse : intégrité des chiffres, licence, preuve expérimentale, périmètre, interopérabilité, publication, puis communauté.

Quatre principes guident chaque décision :

- **Preuve avant promesse** : aucun chiffre dans le README sans script, carte, commit et date qui le reproduisent.
- **Comparer à la meilleure alternative** : TFLite Micro + CMSIS-NN en int8, sur la même tâche et à précision égale.
- **Un cœur petit et stable** : tout ce qui n'est pas indispensable à l'inférence sort du crate principal.
- **Montrer aussi les défaites** : un résultat défavorable publié vaut mieux qu'un gain contesté.

| Indicateur | Cible | Échéance |
| --- | --- | --- |
| Chiffres non sourcés dans README et docs | 0 | fin phase 0 |
| Licence permissive (MIT OR Apache-2.0) | adoptée | fin phase 1 |
| Baseline TFLM/CMSIS-NN mesurée sur la même carte | 1 tâche minimum | fin phase 2 |
| Jeux de données publics évalués de bout en bout | 3 (GSC, SHD, DVS Gesture) | fin phase 3 |
| Perte de précision après quantification Q16.16 | < 1 point | fin phase 3 |
| Graphes NIR réels importés depuis 3 frameworks | snnTorch, Norse, SpikingJelly | fin phase 5 |
| Version publiée sur crates.io avec API stable | 1.0 | fin du plan |
| Mainteneurs actifs | 2 minimum | fin du plan |

## Phase 0 — Corrections immédiates (semaines 1-2)

Cette phase supprime tout ce qui ferait perdre la confiance d'un relecteur dès la première lecture. Elle ne demande ni matériel ni nouvelle mesure, seulement de la rigueur.

**Statut au 25 octobre 2026 :** tous les points sont cochés (rien n'est encore commité). Les trois derniers ont été faits ainsi :

- Énergie : `HardwareProfile`, `energy_constants` et `EnergyEstimator::estimate_picojoules*` / `estimate_nanojoules_fix32*` sont dépréciés ; `EnergyEstimator::cpu_active_nanojoules(cycles, puissance_uw, horloge_hz)` les remplace (vérifié : 11 972 cycles à 48 MHz et 13,86 mW donnent 3 456 nJ). Les lignes d'énergie sont retirées des démos 05, 06 et 12 et du test NARMA ; `docs/neurobench_evaluation_protocol.md` est corrigé.
- Exemple 09 : LIF dense, LIF paresseux et BPTN résolvent la même tâche synthétique (rafale faible sur 4 canaux parmi 16, dans les 16 derniers ticks d'une fenêtre de 64, 20 000 fenêtres) avec une décision par fenêtre. Résultats sur Apple M1 Pro : LIF 90,84 %, BPTN 93,44 % ; BPTN environ 1,25× plus rapide avec une évaluation par fenêtre et 0,95× avec une évaluation à chaque tick ; 16 B / 9 B (LIF) contre 260 B / 128 B (BPTN) en Flash / SRAM. Paramètres choisis par recherche en grille sur les mêmes données, sans jeu de test séparé. Les anciens ratios (14 à 27×, 2,7 à 4,9×) sont retirés.
- Firmware : « KEYWORD DETECTED » était déjà corrigé dans `bptn_benchmark.rs` ; la suite 4 de `silicon_test_runner.rs` (« KWS sur signaux acoustiques réels ») est renommée en test fonctionnel sur motifs de bits écrits à la main.

**Chiffres faux ou non mesurés**

- [x] Corriger l'unité d'énergie dans `examples/wio_e5_benchmark/src/bin/bptn_benchmark.rs` (ligne 235) : `(us / 1000) * mW` donne des µJ. Renommer la variable en `bptn_energy_uj` et corriger les 2,37 nJ en 2,37 µJ dans le README, `docs/wio_e5_silicon_benchmark_report.md` et le whitepaper.
- [x] Supprimer les valeurs codées en dur (18 500 et 58 000 cycles, ~5,34 et ~16,75 nJ, 144 et 850 octets) et les ratios 2,3× / 7,1× qui en découlent. Les remplacer par « non mesuré — voir phase 2 ».
- [x] Réécrire l'encadré du README pour montrer les deux régimes : en streaming, le LIF est 2,7 à 4,9× plus rapide que BPTN ; en mode bloc, BPTN gagne mais coûte 4× plus de Flash. *(Remplacé depuis par les résultats de l'exemple 09 sur tâche commune.)*
- [x] Retirer le tableau d'énergie fondé sur les constantes ASIC (0,1 pJ par AC, 3,1 pJ par MAC). Sur MCU, calculer l'énergie comme cycles × puissance active / fréquence, puis la mesurer en phase 2.
- [x] Supprimer l'affirmation « 10 ans sur CR2032 », ou la recalculer avec le budget complet : courant de veille du MCU, réveils, capteur et acquisition.
- [x] Corriger « unbiased round-to-nearest » dans `numeric.rs` : soit implémenter l'arrondi au pair, soit décrire l'arrondi comme « au demi supérieur ».
- [x] Rendre l'exemple 09 comparable : même tâche, même sortie attendue et précision reportée (avant : LIF émettait 2,2 M de spikes et BPTN 102 M).

**Annonces sans contenu**

- [x] Retirer l'instruction `snn_core = "0.4.0"` tant que le crate n'est pas publié ; documenter l'usage par `git = "..."`.
- [x] Retirer « Official managed component » (ESP-IDF), le CMSIS-Pack et la « MLPerf Tiny submission specification » du README. Les déplacer dans une section Roadmap.
- [x] Renommer `05_shd_cochlea_benchmark` en `05_synthetic_cochlea_demo` et préciser partout que les données sont synthétiques.
- [x] Retirer la démo « KEYWORD DETECTED » du firmware de benchmark : elle teste un bit, pas un mot-clé.

**Exemples et ton**

- [x] Réparer l'exemple `ModelDescriptor` du README : `l1_syn` et `l1_cfg` doivent être des items `static` ou `const`, ou tout passer en `let`.
- [x] Compiler le README dans les tests : `#![doc = include_str!("../README.md")]` dans `lib.rs` transforme chaque bloc Rust en doctest.
- [x] Retirer emojis et superlatifs (« killer », « industrial », « unbiased », « 100% executed »). Renommer l'exemple 12 en `12_can_ids_benchmark`.
- [x] Créer `CLAIMS.md` : une ligne par chiffre public, avec script, carte, commit et date de mesure.
- [x] Ajouter au CHANGELOG une entrée « Corrections » qui liste les chiffres retirés ou corrigés.

**Critère de sortie** : `cargo test` compile tous les exemples du README, et chaque chiffre du README renvoie à une ligne de `CLAIMS.md`.

## Phase 1 — Licence et gouvernance (semaines 1-3)

Recommandation : passer en double licence **MIT OR Apache-2.0**, la norme de l'écosystème Rust embarqué. Tu es le seul auteur de tous les commits : tu peux donc changer de licence aujourd'hui, sans l'accord de personne. Cela deviendra difficile dès qu'il y aura des contributeurs externes.

| Option | Adoption industrielle | Monétisation possible | Verdict |
| --- | --- | --- | --- |
| MIT OR Apache-2.0 | maximale, norme Rust | support, portages, formation, outils pro | recommandée |
| Apache-2.0 seule | très bonne, clause brevets | idem | acceptable |
| MPL-2.0 | bonne, copyleft fichier par fichier | idem, plus les modifications reviennent | possible |
| PolyForm Noncommercial (actuelle) | nulle pour un usage produit | licences commerciales au cas par cas | incompatible avec l'objectif |

Si tu veux garder une part payante, place-la hors du runtime : compilateur avec optimisations avancées, portages sur mesure, support, certification (par exemple IEC 61508 ou ISO 26262 pour l'automobile).

- [ ] Choisir la licence et remplacer `LICENSE.md` par `LICENSE-MIT` et `LICENSE-APACHE` ; mettre à jour `Cargo.toml`, `snn_core.pdsc`, `idf_component.yml`, `pyproject.toml` et les en-têtes C.
- [ ] Vérifier que le nom n'est pas déjà pris sur crates.io, PyPI et le registre ESP. « snn_core » est générique ; un nom distinctif se retient et se protège mieux.
- [ ] Ajouter `CONTRIBUTING.md` avec signature DCO (`Signed-off-by`), `SECURITY.md` (canal de signalement privé) et `CODE_OF_CONDUCT.md`.
- [ ] Ajouter `MAINTAINERS.md` et une règle simple : tout changement d'API publique après la 1.0 passe par une RFC courte en issue.
- [ ] Supprimer `AGENTS.md` du dépôt public, ou le réécrire en guide de contribution sans affirmations de performance.

## Phase 2 — Protocole de mesure et baselines réelles (novembre-décembre 2026)

Cette phase remplace les estimations par des mesures. Un industriel ne retiendra qu'une comparaison : snn_core face à TFLite Micro + CMSIS-NN, sur la même carte, la même tâche et à précision égale.

**Banc matériel**

- Wio-E5 (STM32WLE5, Cortex-M4 sans FPU) : la carte déjà utilisée.
- Une carte Cortex-M33 basse consommation, par exemple STM32U5 ou nRF5340, cible fréquente des produits récents.
- Une carte RISC-V, par exemple ESP32-C3, pour valider la portabilité.
- Un instrument de mesure de courant : Nordic PPK2 (entrée de gamme) ou Joulescope (plus précis). Le compteur de cycles DWT reste utile mais ne mesure pas l'énergie.

**Baselines à implémenter vraiment**

1. TFLite Micro + CMSIS-NN, modèle int8 de référence (DS-CNN pour le KWS de MLPerf Tiny).
2. Un GRU ou un CNN 1D int8 de taille équivalente, entraîné sur les mêmes données.
3. snn_core en LIF Q16.16, puis en INT8 CSR.
4. BPTN, seulement si une version entraînée existe sur la tâche (phase 7).

**Métriques rapportées pour chaque configuration**

| Métrique | Unité | Comment |
| --- | --- | --- |
| Précision top-1 | % | sur le jeu de test complet, exécuté sur la carte |
| Latence par inférence | µs, médiane et p99 | DWT, 1 000 exécutions minimum |
| Énergie par inférence | µJ | intégrée au PPK2 ou au Joulescope |
| Courant moyen au rapport cyclique cible | µA | veille, réveils et capteur inclus |
| Flash | octets | sortie de `cargo size` ou `arm-none-eabi-size` |
| SRAM au pic | octets | arena plus pile mesurée par motif de remplissage |

**Règles d'équité**

- Même extraction de caractéristiques des deux côtés, et son coût est compté dans les deux cas (FFT ou banc de filtres).
- Mêmes options de compilation : niveau d'optimisation, LTO, fréquence d'horloge, même version de compilateur.
- Comparaison à précision égale, ou courbe précision / énergie quand elles diffèrent.
- Données brutes, scripts et binaires publiés dans `bench/` ; chaque exécution produit un JSON horodaté avec commit et numéro de carte.

- [ ] Acheter le PPK2 et la deuxième carte.
- [ ] Porter le DS-CNN de MLPerf Tiny KWS sur le Wio-E5 avec TFLite Micro + CMSIS-NN.
- [ ] Écrire le harnais : envoi des échantillons par UART ou RTT, collecte des prédictions, calcul de la précision sur l'hôte.
- [ ] Publier le premier rapport de mesure, y compris si snn_core perd.

## Phase 3 — Jeux de données réels et pipeline de bout en bout (janvier-mars 2027)

L'objectif est de démontrer le chemin complet qu'un utilisateur suivra : entraîner en PyTorch, exporter en NIR, compiler, flasher, puis mesurer la précision sur la carte. Aujourd'hui, aucun jeu de données public n'est évalué de cette façon.

| Jeu de données | Tâche | Pourquoi lui | Priorité |
| --- | --- | --- | --- |
| Google Speech Commands v2 | détection de mots-clés | référence KWS de MLPerf Tiny, comparaison directe avec TinyML | 1 |
| Spiking Heidelberg Digits (SHD) | chiffres parlés, entrée en spikes | référence de la communauté SNN, résultats publiés nombreux | 1 |
| DVS128 Gesture | gestes, caméra événementielle | cas où l'entrée est déjà événementielle, là où un SNN a le plus de chances de gagner | 2 |
| MLPerf Tiny Anomaly Detection (ToyADMOS) | détection d'anomalies | correspond au récit « sentinelle » du README | 2 |
| Car Hacking Dataset (complet) | détection d'intrusion CAN | prolonge l'exemple 12 avec le jeu entier, pas un extrait | 3 |

**Pipeline à construire**

1. Entraînement avec snnTorch (gradient de substitution), recettes et hyperparamètres versionnés.
2. Export NIR avec la fonction officielle du framework.
3. Compilation par `snn-compile` en Q16.16 et en INT8.
4. Flash sur carte, puis évaluation du jeu de test complet via le harnais de la phase 2.
5. Rapport : précision en flottant, en Q16.16 et en INT8 ; latence, énergie, Flash et SRAM ; même tableau pour la baseline TinyML.

- [ ] Créer un dépôt séparé `snn_core-models` pour les recettes d'entraînement, les poids et les scripts d'évaluation.
- [ ] Viser une perte de précision inférieure à 1 point entre le flottant et la carte ; sinon, documenter la cause (quantification, approximation de la décroissance, saturation).
- [ ] Comparer les résultats SHD et DVS Gesture aux résultats publiés par d'autres travaux, avec la référence citée.
- [ ] Soumettre les résultats au volet algorithmique de NeuroBench.

## Phase 4 — Périmètre et architecture (décembre 2026-février 2027)

Le crate actuel mélange un runtime, de la recherche (BPTN), de l'apprentissage, des applications (parseur CAN, NARMA, DWT) et deux voies C. Il faut le découper en un workspace où seul le runtime est critique et stable.

**Statut au 25 octobre 2026 :** découpage en workspace fait (runtime `snn_core`, `snn_core_bptn`, `snn_core_c`, `demos`), avec un découpage plus simple que la table ci-dessous : les neurones et l'apprentissage restent dans le runtime. L'en-tête C n'est pas généré par cbindgen (indisponible hors ligne) : un test (`header_parity.rs`) échoue si `snn_core.h` diverge du Rust, et la CI compile un test C contre la bibliothèque. Le runtime C99 écrit à la main est supprimé. Fait aussi : licence MIT, audit des `panic` (4 `assert!` restants, documentés), commentaires `SAFETY` sur l'assembleur et les conversions de buffers. Fait ensuite : `ModelDescriptor` documenté comme API unique d'inférence (`ChainedSpikingNetwork`, `ArenaSpikingNetwork` et leur macro dépréciés ; `GenericStaticNetwork` gardé comme simulateur avec apprentissage), et format binaire versionné `.snnb` 1.0 (magic, version, CRC-32, lecture en place, fichier de référence Rust/Python). Fait le 25 octobre : le firmware Wio-E5 et le runtime bare-metal passent `cargo check --target thumbv7em-none-eabi` sur une machine ARM (Apple M1 Pro), après deux corrections : une table `[workspace]` vide dans `examples/wio_e5_benchmark/Cargo.toml`, et `#![no_std]` ajouté à `snn_core_bptn`, qui ne le déclarait pas. La sonde ST-Link V2 voit la puce par SWD (port de debug STM32WL, référence 0x497). Fait aussi le 25 octobre : les 4 firmwares sont flashés et exécutés sur la carte. Les tests ont montré que la carte tournait à 4 MHz (horloge de reset) et non à 48 MHz, et que plusieurs mesures étaient précalculées par le compilateur. Le firmware règle maintenant 48 MHz avec 2 états d'attente, relit l'horloge, et protège ses mesures avec `black_box`. La couche BPTN mesure 11 972 cycles (249,42 µs), contre 8 193 cycles et 170,69 µs annoncés auparavant ; `silicon_test_runner` passe 20 tests sur 20. Reste : commentaire `SAFETY` par bloc FFI, et mesure du courant (phase 2).

| Crate cible | Contenu | Stabilité visée |
| --- | --- | --- |
| `snn-core` | Fix32/Fix16, bitsets, CSR en ROM, LIF et CUBA-LIF, `ModelDescriptor` + `ExecutionContext`, arena, intrinsics DSP sous feature | 1.0, `no_std`, sans `alloc` |
| `snn-core-neurons` | ALIF, Izhikevich, populations paresseuses | 0.x |
| `snn-core-learn` | STDP, R-STDP, readouts entraînables | 0.x, `alloc` autorisé |
| `snn-core-ffi` | ABI C générée par cbindgen, bibliothèque statique | suit `snn-core` |
| `bptn` | moteur BPTN complet | expérimental |
| dépôt `snn_core-examples` | parseur CAN, NARMA, DWT, démos | hors versionnement |
| paquet Python `snn-compile` | un seul compilateur : fusion de `nir_to_snn_core.py` dans `tools/snn_compiler` | 0.x, publié sur PyPI |

**Une seule API publique d'inférence**

- [x] Faire de `ModelDescriptor` + `ExecutionContext` l'unique point d'entrée documenté.
- [ ] Rendre internes ou déprécier `GenericStaticNetwork`, `ChainedSpikingNetwork`, `ArenaSpikingNetwork` et la macro `arena_spiking_network!`.
- [x] Documenter le modèle mémoire : ce qui va en Flash, ce qui va en SRAM, l'alignement, et la taille d'arena calculée par le compilateur.

**Une seule source de vérité pour le C**

Aujourd'hui, `snn_core_c99.h` (1 305 lignes) réimplémente le moteur en parallèle de la FFI Rust. Deux options :

1. **Recommandée** : supprimer le runtime C99 écrit à la main et livrer `libsnn_core.a` avec un en-tête généré par cbindgen.
2. Garder le runtime C99 seulement si des clients C purs le demandent. Dans ce cas, des vecteurs de test produits par Rust (JSON) doivent être rejoués par la suite C en CI, avec égalité bit à bit exigée.

**Robustesse du chemin d'inférence**

- [x] Auditer les 23 `unwrap`, `expect`, `panic!` et `unreachable!` hors tests dans `src/` : aucun ne doit rester sur le chemin d'inférence.
- [ ] Vérifier pointeurs nuls, longueurs et alignement à chaque fonction FFI (167 blocs `unsafe` dans `ffi.rs`) ; chaque bloc porte un commentaire `// SAFETY:`.
- [x] Versionner le format binaire en Flash : magic, version, CRC, refus explicite d'un format inconnu.

## Phase 5 — Interopérabilité NIR complète (février-avril 2027)

NIR est le seul standard d'échange accepté dans l'écosystème SNN. Être un backend NIR complet et testé est le moyen le plus direct d'exister aux yeux des chercheurs et des industriels. Aujourd'hui, le chargeur du CLI ne gère que `Affine`, `Linear`, `LIF`, `CubaLIF` et `IF`, et ramène `tau` et le seuil à un scalaire.

- [ ] Conserver les paramètres par neurone (`tau`, `v_threshold`, `r`, `v_leak`) sous forme de tableaux, avec une option de factorisation quand ils sont uniformes.
- [ ] Ajouter les nœuds manquants : `Conv1d`, `Conv2d`, `SumPool2d`, `AvgPool2d`, `Flatten`, `Scale`, `Threshold`, `Delay`, `Input`, `Output`.
- [ ] Gérer les sous-graphes récurrents (cycles dans le graphe NIR).
- [ ] Tester en CI des graphes réellement exportés par snnTorch, Norse, SpikingJelly, Sinabs et Rockpool, en versions figées. Remplacer le test actuel `--input synthetic`.
- [ ] Comparer chaque graphe importé à une simulation flottante de référence, avec une tolérance déclarée sur les trains de spikes et les sorties.
- [ ] Publier une matrice de compatibilité : nœud NIR, support (complet, partiel, absent), limites connues.
- [ ] Proposer à l'équipe NIR de référencer snn_core comme backend, une fois la matrice publiée et la CI verte.
- [ ] Reléguer l'import ONNX au statut expérimental : ONNX n'a pas de représentation standard des neurones à spikes, chaque export est un cas particulier.

## Phase 6 — Publication, qualité et release engineering (novembre 2026-juin 2027)

Publier tôt une 0.5 honnête, puis durcir jusqu'à une 1.0 dont l'API ne bouge plus. Rien n'est annoncé comme disponible avant d'être réellement publié et testé.

**Publication**

- [ ] Publier la 0.5.0 sur crates.io dès la fin des phases 0 et 1, avec docs.rs et toutes les features documentées.
- [ ] Activer `#![deny(missing_docs)]` sur l'API publique du crate runtime.
- [ ] Déclarer la MSRV (l'édition 2024 impose Rust 1.85 minimum) et la tester en CI.
- [ ] Publier le module Zephyr, le composant ESP-IDF et le CMSIS-Pack uniquement après un build CI réussi sur une carte de chaque écosystème (`west build`, `idf.py build`, `packchk`).
- [ ] Tenir un CHANGELOG au format Keep a Changelog et taguer chaque release.

**Qualité vérifiée en CI**

| Contrôle | Outil | Fréquence |
| --- | --- | --- |
| Compilation croisée thumbv6m, thumbv7em, thumbv8m.main, riscv32imc | cargo check / build | chaque PR |
| Rupture d'API | cargo-semver-checks | chaque PR |
| Taille binaire de l'exemple de référence | cargo-bloat, seuil d'alerte | chaque PR |
| Comportement indéfini dans la FFI et les blocs `unsafe` | Miri | chaque PR |
| Fuzzing du chargeur binaire, du chargeur NIR et de `snn_model_step` | cargo-fuzz | nuit |
| Couverture de tests du runtime | cargo-llvm-cov, cible 80 % | chaque PR |
| Tests sur carte réelle (Wio-E5) | probe-rs sur un runner auto-hébergé | nuit |
| Benchmarks de performance et détection de régression | harnais de la phase 2 | hebdomadaire |

- [ ] Rendre `cargo test` exécutable hors ligne et sans dépendances lourdes : isoler criterion et nalgebra dans un crate de benchs, retirer `fastrand` des tests unitaires de `narma.rs`.
- [ ] Geler l'API pour la 1.0 après 2 mois sans changement cassant sur les 0.x.

## Phase 7 — BPTN comme piste de recherche (en parallèle, article visé au printemps 2027)

BPTN sort du produit et devient un projet de recherche avec son propre crate. Il ne revient dans le cœur que s'il bat le LIF ou le TinyML int8 sur une tâche réelle, à précision égale.

**Premier résultat (25 octobre 2026, hôte) :** sur SHD, avec 64 neurones cachés, la même entrée (64 canaux × 64 pas) et la même lecture int8, BPTN entraîné par STE atteint 68,7 % (3 graines), contre 69,7 % pour un réseau binarisé de même taille, et environ 60 % pour les LIF de référence (non optimisés). BPTN occupe 2× plus de Flash que le réseau binarisé (67 Ko contre 34 Ko) pour une précision et un coût estimé équivalents. Script : `tools/bptn_torch/train_shd_comparison.py`.

**Neurone LUT temporel (26 octobre 2026) :** chaque neurone lit 6 bits de l'historique et sort le bit d'une table apprise. Sur le Wio-E5, il coûte 49,7 cycles par LUT en code généré (77,7 avec une table d'adresses). Sur SHD, avec des connexions aléatoires fixes, il atteint au mieux 64,2 % (4 000 LUT, environ 215 k cycles estimés), contre environ 69 % pour BPTN et le réseau binarisé au même coût. Il ne fait pas mieux en l'état ; la piste restante est l'apprentissage des connexions.

**Neurone LUT avec connexions apprises (26 octobre 2026) :** en apprenant quels bits chaque LUT lit (argmax avec estimateur straight-through, comme les Differentiable Weightless Neural Networks), le réseau LUT atteint 67,6 % (1 000 LUT), 71,4 % (2 000) et 75,0 % (4 000) sur SHD (3 graines). **Mesuré sur la carte** avec les poids entraînés, sur les 2 264 échantillons de test : 2 000 LUT donnent 71,95 % en 223 791 cycles (4,66 ms) et 40,4 Ko ; un réseau binarisé de 100 neurones donne 71,69 % en 216 758 cycles (4,52 ms) et 55,1 Ko. Les prédictions sur la carte sont identiques à l'hôte pour les deux. À cycles égaux, aucune différence significative ; à précision égale, le réseau LUT prend environ 25 % de Flash en moins. Les avantages annoncés plus tôt (5 à 12 points, puis 2 à 3 points) venaient d'estimations de coût et sont retirés. Reste : un réseau int8 CMSIS-NN comme référence, et d'autres jeux de données.

**Lire moins de modèle par inférence (26 octobre 2026) :** sur la carte, le coût suit les octets de modèle lus (environ 4 à 5,5 cycles par octet). Deux pistes mesurées sur le réseau de 2 000 LUT : l'arrêt anticipé du group-sum (marge de 7 votes) réduit le coût moyen de 32 % pour −0,4 point (71,55 %) ; le streaming incrémental, qui ne recalcule que les LUT dont une entrée a changé, réduit le coût par tick de 38 % (139 k contre 224 k cycles) en restant exact. Aucune des deux n'atteint le critère de 2×, et les deux ont un pire cas plus lent que l'évaluation complète. Pistes suivantes : vérifier plus rarement la condition d'arrêt, alléger la tenue des dépendances, et les programmes de branchement appris.

**Programmes de branchement appris (26 octobre 2026) :** une forêt d'arbres de décision dont chaque nœud est une LUT (6 bits d'entrée appris, sortie qui choisit l'enfant) et dont les feuilles portent 20 scores int8. Une inférence ne lit que quelques nœuds par arbre. **Mesuré sur la carte**, sur les 2 264 échantillons de test, avec des prédictions identiques à l'hôte : 64 arbres de profondeur 4 donnent 74,82 % en 30 124 cycles (0,63 ms) et 41,7 Ko ; 256 arbres de profondeur 2 donnent 77,30 % en 78 914 cycles. Sur 3 graines, 64 × 4 fait 74,31 % en moyenne, contre 71,72 % pour le meilleur réseau binarisé mesuré (100 neurones, 216 758 cycles, 55,1 Ko) : +2,6 points avec 7,2× moins de cycles et 24 % de Flash en moins. **C'est le premier résultat qui dépasse le critère de 2× sur une vraie tâche, mesuré sur silicium.** **Comparaison avec les arbres classiques (même entrée, scikit-learn) :** un random forest ne tient pas en Flash à une précision utile (71,6 % demande 11 Mo). Le gradient boosting fait 72,04 % sur la carte en 146 371 cycles et 121,1 Ko (50 itérations, profondeur 4), et n'atteint 75 % qu'avec 8 000 nœuds et 237 Ko. La forêt d'arbres LUT fait donc +2,8 points avec 4,9× moins de cycles et 2,9× moins de Flash que le meilleur arbre classique qui tient sur la carte : le gain vient bien des nœuds LUT à entrées apprises, pas seulement du fait d'utiliser des arbres. À vérifier : un réseau int8 CMSIS-NN, d'autres jeux de données, et un réglage plus poussé du gradient boosting.

**Référence int8 CMSIS-NN (26 octobre 2026, étape 1 du plan d'avancement) :** un CNN 1D int8, quantifié pour CMSIS-NN, réglé sur la validation comme les autres modèles (3 graines), atteint 66,6 % (35 k MAC, 5 Ko), 70,0 % (134 k MAC) et 74,4 % (1,12 M MAC, 47 Ko). À précision égale (~74 %), il demande au moins 562 k cycles même au débit maximal de `SMLAD` (2 MAC par cycle), contre 30 124 cycles mesurés pour la forêt LUT, soit au moins 18× plus. En revanche, à précision plus basse, il est bien plus compact en Flash (5 Ko pour 66,6 %). CMSIS-NN est compilé dans le firmware (`cnn_shd_eval`) ; la mesure sur la carte reste à faire. Avec le même protocole de réglage, la forêt fait 73,8 % et le gradient boosting 73,1 %.

**Étape 2 du plan d'avancement (26 octobre 2026, hôte, graine 0) : critère non atteint.** Sur Google Speech Commands v2 (35 mots) et sur le jeu CAN AutoHack, encodés eux aussi en 64 × 64 bits, la forêt LUT perd contre le plus petit CNN int8 : 56,1 % contre 61,3 % (Speech Commands, meilleure forêt 256 × 2 contre CNN tiny) et 74,4 % contre 77,8 % (AutoHack), alors que ce CNN coûte moins de cycles (au moins 17,8 k) et 8 à 10× moins de Flash. Les forêts sous-apprennent sur ces tâches. Cause probable : l'information y est agrégée sur toute la fenêtre (comptages, formes spectrales), ce que la moyenne globale d'un CNN capte et des nœuds de 6 bits non. L'avantage mesuré sur SHD est donc propre à cette tâche à ce stade. Décision proposée : ne pas passer à l'étape 3 (intégration au runtime) tant qu'une variante de la forêt ne gagne pas sur au moins une seconde tâche.

**Forêt hybride (26 octobre 2026) :** une couche d'agrégation bon marché (comptages de bits par canal sur 4 fenêtres et sur toute la fenêtre, codés en thermomètre) placée avant la forêt corrige le sous-apprentissage sur le CAN : 78,1 % pour la forêt 256 × 2, au niveau du petit CNN int8 (77,8 %). Mais le gradient boosting, avec la même entrée, atteint aussi 78,1 % pour environ 2× moins de cycles. Sur Speech Commands, l'agrégation n'aide pas (56,1 %, contre 61,3 % pour le petit CNN). **Conclusion de l'étape 2 : la forêt LUT n'a d'avantage mesuré que sur SHD ; sur les deux autres tâches, elle fait au mieux jeu égal avec des méthodes classiques (CNN int8, gradient boosting) sans les battre en coût.**

**Entrées multi-échelles (27 octobre 2026, hôte, graine 0) :** au lieu de comptages fixés à la main, une banque de 4 788 comptages (groupes de 1, 4 ou 16 canaux voisins × fenêtres de 4 à 64 pas, codés en thermomètre sur 3 bits) dans laquelle les connexions apprises choisissent ce que lit chaque nœud. Pour la forêt, le gain est faible : 57,9 % sur Speech Commands (256 × 2, contre 56,1 % avec l'entrée hybride, toujours sous les 61,3 % du petit CNN) et aucun gain sur AutoHack (76,9 % en 64 × 4, 76,5 % en 256 × 2). Le gradient boosting en profite davantage sur AutoHack : 80,0 %, le meilleur résultat sur cette tâche, 2,3 points au-dessus du petit CNN. **Conclusion : ce ne sont pas les entrées qui limitent la forêt, mais son apprentissage (seuils et partitions appris par descente de gradient).** Le coût de la banque sur la carte (image intégrale) n'est pas mesuré.

- [ ] Faire une revue de littérature et situer BPTN face aux réseaux binarisés, au calcul hyperdimensionnel, aux convolutions temporelles à noyaux binaires et aux machines de Tsetlin. Ce qui est vraiment nouveau doit pouvoir s'énoncer en une phrase.
- [ ] Définir une méthode d'entraînement qui passe à l'échelle (par exemple un estimateur straight-through sur les masques), au-delà de la démo XOR temporel.
- [ ] Évaluer sur Google Speech Commands et SHD avec le protocole de la phase 2 : précision, énergie mesurée et Flash, en streaming comme en mode bloc.
- [ ] Rédiger un article et le soumettre à un atelier relu par des pairs (tinyML Research Symposium, ICONS ou NICE).
- [ ] **Critère d'arrêt** : si BPTN reste moins précis ou plus coûteux en Flash sans gain d'énergie mesuré d'au moins 2×, l'arrêter ou le laisser en expérimental sans le mettre en avant.

## Phase 8 — Communauté et adoption (continu, dès la 0.5)

Un standard est un projet que d'autres utilisent et maintiennent. Le bus factor est aujourd'hui de 1 et il n'y a aucun utilisateur externe connu.

**Publics visés, dans l'ordre**

1. Les chercheurs SNN qui veulent déployer un modèle snnTorch ou Norse sur un vrai MCU.
2. Les équipes embarquées qui explorent les capteurs événementiels (caméras DVS, cochlées silicium, bus CAN).
3. Les fabricants de MCU et de capteurs, comme partenaires de portage.

**Actions**

- [ ] Publier un article de blog avec les résultats honnêtes de la phase 3, victoires et défaites comprises.
- [ ] Présenter le projet à la communauté Open Neuromorphic, sur le forum tinyML et au groupe de travail Rust Embedded.
- [ ] Étiqueter 10 issues « good first issue » et répondre à chaque issue sous 7 jours.
- [ ] Trouver un co-mainteneur, idéalement dans un laboratoire qui utilise déjà snnTorch ou NIR.
- [ ] Livrer 2 ou 3 designs de référence complets (KWS sur Wio-E5, gestes DVS, IDS CAN), du capteur à la décision.
- [ ] Suivre l'adoption : téléchargements crates.io et PyPI, contributeurs externes, projets dépendants, citations.

## Calendrier, jalons et critères go/no-go

Le plan dure 9 mois, du 26 octobre 2026 à fin juillet 2027. La porte G2, fin mars 2027, est décisive : elle dit si snn_core apporte un gain mesuré sur une tâche réelle.

*(Le document d'origine contient ici un calendrier graphique : 9 phases, 4 portes.)*

Les phases 0 à 2 sont séquentielles ; la publication, BPTN et la communauté avancent en parallèle dès novembre.

| Porte | Date | Critère de passage | Si non atteint |
| --- | --- | --- | --- |
| G0 | 15 nov. 2026 | phase 0 terminée, licence changée, exemples du README compilés en doctests | pas de publication 0.5 |
| G1 | 20 déc. 2026 | TFLM + CMSIS-NN et snn_core mesurés au wattmètre sur la même carte et la même tâche | prolonger la phase 2, ne pas lancer la phase 3 |
| G2 | 31 mars 2027 | sur au moins un jeu public, énergie par inférence au moins 2× inférieure à CMSIS-NN, à 1 point de précision près | repositionner en runtime NIR de prototypage et publier le résultat |
| G3 | 30 juin 2027 | import NIR testé en CI pour 3 frameworks, API sans changement cassant depuis 2 mois, 2 mainteneurs | repousser la 1.0 |

Hypothèse de capacité : environ un développeur à mi-temps. À plein temps, le même plan tient en 4 à 5 mois.

## Risques et parades

Le risque principal est que les mesures honnêtes montrent qu'un SNN sur CPU généraliste ne bat pas CMSIS-NN. Le plan est conçu pour le découvrir tôt (portes G1 et G2) plutôt qu'après 9 mois.

| Risque | Probabilité | Impact | Parade |
| --- | --- | --- | --- |
| Aucun gain mesuré face à TFLM/CMSIS-NN sur tâche réelle | élevée | élevé | repositionner en runtime NIR pour MCU (prototypage, recherche, capteurs événementiels), publier quand même les résultats |
| Un seul mainteneur, projet bloqué en cas d'absence | élevée | élevé | co-mainteneur avant la 1.0, documentation d'architecture, CI qui ne dépend pas d'une personne |
| Le matériel neuromorphique dédié (Xylo, Akida, Loihi) capte l'argument énergie | moyenne | moyen | se positionner en exécution logicielle de référence et de repli, compatible NIR avec ces puces |
| NIR évolue et casse l'import | moyenne | moyen | versions figées en CI, contributions en amont, matrice de compatibilité par version |
| La licence permissive réduit les revenus possibles | moyenne | faible | services, portages, support et certification en offre payante |
| Le périmètre regonfle | moyenne | moyen | règle écrite : tout ce qui n'est pas indispensable à l'inférence va dans un crate séparé ou dans le dépôt d'exemples |
| Crédibilité déjà entamée par l'ancien README | faible | moyen | entrée « Corrections » publique dans le CHANGELOG et `CLAIMS.md` |

## Sources

- [NeuroBench (GitHub)](https://github.com/neurobench/neurobench)
- [NIR — Neuromorphic Intermediate Representation](https://nnir.readthedocs.io/en/latest/)
- [CARLsim sur microcontrôleurs, arXiv 2604.16474](https://arxiv.org/pdf/2604.16474)
- [Compression and Inference of SNNs on Resource-Constrained Hardware, arXiv 2511.12136](https://arxiv.org/pdf/2511.12136)
- Analyse critique de snn_core du 25 octobre 2026 (commit `acc183b`), réalisée dans cette conversation.
