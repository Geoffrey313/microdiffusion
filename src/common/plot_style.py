"""Shared publication style for empirical figures - THE SINGLE SOURCE OF TRUTH.

All font sizes / LaTeX (mathtext) settings live here and are applied via rcParams in setup_mpl(). Plot scripts
MUST NOT pass per-call `fontsize=`/`labelsize=` overrides - they inherit these defaults so every figure matches
(the reference look is intraday_reynolds.png). If a one-off size is unavoidable, use the FS_* constants below so
it is still centrally defined. To restyle every figure, change the numbers here and regenerate.
"""
from __future__ import annotations

import os
import sys
import tempfile

# --------------------------------------------------------------------------- #
#  Language layer (backward compatible).
#
#  Every user-facing text string in the figure-emitting scripts is wrapped with
#  T(...). When MICRODIFFUSION_LANG is unset or "en", T(s) returns s UNCHANGED,
#  so the English figures and the whole English reproduction are byte-for-byte
#  unaffected. When MICRODIFFUSION_LANG == "fr", T(s) returns the French phrase
#  from _FR (falling back to s, with a one-time stderr note, if a phrase is
#  missing). All mathematics ($...$) is kept verbatim inside the French values.
# --------------------------------------------------------------------------- #
_LANG = os.environ.get("MICRODIFFUSION_LANG", "en")

# English label phrase -> French. Keys are the exact strings passed to T() in the
# figure scripts; the math (anything inside $...$) is identical on both sides.
_FR = {
    # --- shared axis vocabulary ------------------------------------------- #
    "best-level imbalance $I$": r"déséquilibre à la meilleure limite $I$",
    "relative spread $S$": r"écart relatif $S$",
    "imbalance $I$": r"déséquilibre $I$",
    "spread $S$": r"écart $S$",
    "number of instruments": "nombre d'instruments",
    "number of cells": "nombre de cellules",
    "tight": "étroit",
    "mid": "moyen",
    "wide": "large",
    "tick": "pas",
    "low": "faible",
    "high": "élevé",
    "data": "données",
    "data $z$": r"données $z$",
    r"density (log scale)": "densité (échelle log)",
    "density (log)": "densité (échelle log)",
    "density": "densité",
    "best-fit Gaussian": "gaussienne au meilleur ajustement",
    "Student-$t$ (ML)": r"Student-$t$ (MV)",
    r"Gen.\ Hyperbolic (ML)": r"hyperbolique généralisée (MV)",
    r"Gen.\\ Hyperbolic (ML)": r"hyperbolique généralisée (MV)",
    # --- garch_figure.py -------------------------------------------------- #
    "$y=x$ (no improvement)": r"$y=x$ (aucune amélioration)",
    "GARCH out-of-sample log-likelihood / obs.":
        "log-vraisemblance GARCH hors échantillon / obs.",
    r"combined (GARCH $\times$ book state) / obs.":
        r"combiné (GARCH $\times$ état du carnet) / obs.",
    "Combined vs GARCH": "Combiné vs GARCH",
    "above the line": "au-dessus de la ligne",
    r"$c_2=0$ (no weight on $a_{xx}$)": r"$c_2=0$ (aucun poids sur $a_{xx}$)",
    "mean": "moyenne",
    "median": "médiane",
    r"fitted combination weight on $a_{xx}$ ($c_2$)":
        r"poids de combinaison ajusté sur $a_{xx}$ ($c_2$)",
    "Weight on the book state": "Poids sur l'état du carnet",
    # --- surface_heatmap_counts.py ---------------------------------------- #
    "tight\n(S tercile 1)": "étroit\n(S tercile 1)",
    "mid\n(S tercile 2)": "moyen\n(S tercile 2)",
    "wide\n(S tercile 3)": "large\n(S tercile 3)",
    r"Diffusion surface $\log_{10}\,a_{xx}/\mathrm{median}$ (pooled)":
        r"Surface de diffusion $\log_{10}\,a_{xx}/\mathrm{median}$ (poolée)",
    "Total observation count per cell": "Nombre total d'observations par cellule",
    "The diffusion surface as a heatmap, with per-cell occupancy: well-populated cells carry the transferable structure":
        "La surface de diffusion en carte de chaleur, avec l'occupation par cellule : les cellules bien peuplées portent la structure transférable",
    # --- gap_figures.py --------------------------------------------------- #
    r"Conditional drift surface $b_x(I,S)$": r"Surface de dérive conditionnelle $b_x(I,S)$",
    r"$(I,S)$ model": r"modèle $(I,S)$",
    r"$(x,I,S)$ model": r"modèle $(x,I,S)$",
    r"out-of-sample $R^2$": r"$R^2$ hors échantillon",
    r"Adding the level $x$ does not help": r"Ajouter le niveau $x$ n'aide pas",
    "hit-rate": "taux de succès",
    r"Diffusion surface $a_{xx}(I,S)$": r"Surface de diffusion $a_{xx}(I,S)$",
    r"train $a_{xx}(c)$  [bps$^2$]": r"$a_{xx}(c)$ d'entraînement  [bps$^2$]",
    r"held-out $\mathbb{E}[(\Delta x)^2\,|\,c]$  [bps$^2$]":
        r"$\mathbb{E}[(\Delta x)^2\,|\,c]$ de test  [bps$^2$]",
    r"Out-of-sample transfer (Spearman": r"Transfert hors échantillon (Spearman",
    # --- us_appendix_figures.py / crypto_appendix_figures.py -------------- #
    r"training cell diffusion $a_{xx}^{\mathrm{train}}$ (fold-normalised)":
        r"diffusion de cellule d'entraînement $a_{xx}^{\mathrm{train}}$ (normalisée par pli)",
    r"held-out cell diffusion $a_{xx}^{\mathrm{test}}$ (fold-normalised)":
        r"diffusion de cellule de test $a_{xx}^{\mathrm{test}}$ (normalisée par pli)",
    "Walk-forward transfer per cell (fold-averaged $\\rho=%.2f$)":
        "Transfert glissant par cellule ($\\rho=%.2f$ moyenné par pli)",
    "The diffusion surface on the United States large-cap panel and its walk-forward transfer":
        "La surface de diffusion sur le panel de grandes capitalisations américaines et son transfert glissant",
    "The diffusion surface on the cryptocurrency panel and its walk-forward transfer":
        "La surface de diffusion sur le panel de cryptomonnaies et son transfert glissant",
    r"standardised increment $z=(\Delta x-b_x)/\sqrt{a_{xx}}$":
        r"incrément standardisé $z=(\Delta x-b_x)/\sqrt{a_{xx}}$",
    "Shape of the kick: Gaussian vs heavy-tailed":
        "Forme du choc : gaussienne vs queue épaisse",
    r"threshold $k$ (in units of $\sqrt{a_{xx}}$)":
        r"seuil $k$ (en unités de $\sqrt{a_{xx}}$)",
    "Tail exceedance on the held-out blocks":
        "Dépassement de queue sur les blocs de test",
    "Heavy-tailed innovation on the United States large-cap panel: standardised held-out increments vs fitted laws":
        "Innovation à queue épaisse sur le panel de grandes capitalisations américaines : incréments de test standardisés vs lois ajustées",
    "Heavy-tailed innovation on the cryptocurrency panel: standardised held-out increments vs fitted laws":
        "Innovation à queue épaisse sur le panel de cryptomonnaies : incréments de test standardisés vs lois ajustées",
    # --- price_band_series.py --------------------------------------------- #
    r"GH heavy-tail extension (99\%)": r"extension de queue épaisse GH (99\%)",
    r"$a_{xx}(I,S)$ scale, Gaussian (99\%)": r"échelle $a_{xx}(I,S)$, gaussienne (99\%)",
    r"model one-step price estimate $x_t+b_x$":
        r"estimation de prix du modèle à un pas $x_t+b_x$",
    "realised mid-price": "prix médian réalisé",
    "mid-price": "prix médian",
    "Full held-out session (": "Session de test complète (",
    "; shaded region $=$ zoom in panel (b)":
        r"; zone ombrée $=$ zoom au panneau (b)",
    "move beyond the Gaussian band": "mouvement au-delà de la bande gaussienne",
    "event index within the held-out session":
        "indice d'événement dans la session de test",
    r"Zoom: the band breathes with $a_{xx}(I,S)$ and is widened by the GH tail":
        r"Zoom : la bande respire avec $a_{xx}(I,S)$ et est élargie par la queue GH",
    "One-step-ahead predictive interval for the price from the state-dependent SDE":
        "Intervalle prédictif à un pas pour le prix issu de l'EDS dépendant de l'état",
    "realised standardised returns": "rendements standardisés réalisés",
    "model: GH innovation": "modèle : innovation GH",
    "Gaussian": "gaussienne",
    r"standardised one-step return $(\Delta x-b_x)/\sqrt{a_{xx}}$":
        r"rendement standardisé à un pas $(\Delta x-b_x)/\sqrt{a_{xx}}$",
    "Return distribution: model vs realised":
        "Distribution des rendements : modèle vs réalisé",
    "realised variance (by state)": "variance réalisée (par état)",
    r"model variance $a_{xx}(I,S)$": r"variance du modèle $a_{xx}(I,S)$",
    r"conditional variance of the one-step move  [bps$^2$]":
        r"variance conditionnelle du mouvement à un pas  [bps$^2$]",
    "Variance distribution along the realised path: model vs realised":
        "Distribution de la variance le long de la trajectoire réalisée : modèle vs réalisé",
    "Model vs realised on held-out data (":
        "Modèle vs réalisé sur données de test (",
    "): the distribution of one-step returns and of the conditional variance":
        ") : la distribution des rendements à un pas et de la variance conditionnelle",
    # --- garch_grid.py ---------------------------------------------------- #
    "GARCH + Gaussian": "GARCH + gaussienne",
    r"GARCH + Student-$t$": r"GARCH + Student-$t$",
    r"$a_{xx}(I,S)$ + GH": r"$a_{xx}(I,S)$ + GH",
    "Combined + GH": "Combiné + GH",
    "perfect calibration": "calibration parfaite",
    "nominal coverage of the predictive interval":
        "couverture nominale de l'intervalle prédictif",
    "empirical coverage (held-out)": "couverture empirique (test)",
    "Are the predictive intervals calibrated?":
        "Les intervalles prédictifs sont-ils calibrés ?",
    r"perfect (ratio $=1$)": r"parfait (ratio $=1$)",
    r"nominal two-sided tail level (\%)": r"niveau de queue bilatéral nominal (\%)",
    "observed / nominal tail exceedance": "dépassement de queue observé / nominal",
    r"Tail-risk calibration (lower tail levels $\rightarrow$)":
        r"Calibration du risque de queue (niveaux de queue plus faibles $\rightarrow$)",
    "Out-of-sample predictive calibration: heavy-tailed innovations improve tail-risk calibration":
        "Calibration prédictive hors échantillon : les innovations à queue épaisse améliorent la calibration du risque de queue",
    # --- zeromove.py ------------------------------------------------------ #
    r"median ${:.2f}$": r"médiane ${:.2f}$",
    r"held-out zero-move share $1-q(I,S)$ per cell":
        r"part de mouvements nuls de test $1-q(I,S)$ par cellule",
    "A large fraction of updates leave the mid unchanged":
        "Une grande part des mises à jour laissent le médian inchangé",
    r"data $z_{\rm full}$ (with zeros)": r"données $z_{\rm full}$ (avec zéros)",
    r"data $z_{+}$ (nonzero, correct scale)":
        r"données $z_{+}$ (non nuls, échelle correcte)",
    r"GH fit on $z_{+}$ (OOS)": r"ajustement GH sur $z_{+}$ (hors échantillon)",
    r"Gaussian fit on $z_{+}$": r"ajustement gaussien sur $z_{+}$",
    r"threshold $k$ (local std units)": r"seuil $k$ (unités d'écart-type local)",
    r"$P(|z| > k)$ on held-out block": r"$P(|z| > k)$ sur le bloc de test",
    "Heavy tail persists on nonzero moves (exkurt $=":
        "La queue épaisse persiste sur les mouvements non nuls (exkurt $=",
    "Zero-move decomposition: the heavy tail is not a zero-inflation artifact --- it survives correct conditional-on-move scaling":
        "Décomposition des mouvements nuls : la queue épaisse n'est pas un artefact d'inflation de zéros --- elle survit à la mise à l'échelle conditionnelle au mouvement correcte",
    # --- regime_transfer.py ----------------------------------------------- #
    r"Calm$\to$Stress": r"Calme$\to$Stress",
    r"Stress$\to$Calm": r"Stress$\to$Calme",
    r"Adjacent\n(baseline)": r"Adjacent\n(référence)",
    r"Adjacent baseline $\rho=":
        r"Référence adjacente $\rho=",
    r"Spearman $\rho$: train $a_{xx}$ vs test $a_{xx}$":
        r"Spearman $\rho$ : $a_{xx}$ d'entraînement vs $a_{xx}$ de test",
    r"$a_{xx}(I,S)$ surface transfer across regimes":
        r"Transfert de la surface $a_{xx}(I,S)$ entre régimes",
    "Adjacent baseline gain$=": "Gain de référence adjacente $=",
    "Mean OOS log-likelihood gain (nats/obs), GH over Gaussian":
        "Gain moyen de log-vraisemblance hors échantillon (nats/obs), GH vs gaussienne",
    "GH tail law: OOS gain across regimes":
        "Loi de queue GH : gain hors échantillon entre régimes",
    r"Cross-regime transfer of $a_{xx}(I,S)$ surface and GH tail law (calm/stress split by cross-instrument median $|\Delta x|$ per trading day)":
        r"Transfert inter-régimes de la surface $a_{xx}(I,S)$ et de la loi de queue GH (partition calme/stress par médiane inter-instruments de $|\Delta x|$ par jour de bourse)",
    # --- var_backtest.py -------------------------------------------------- #
    r"nominal $\alpha{=}0.01$": r"$\alpha{=}0.01$ nominal",
    r"nominal $\alpha{=}0.05$": r"$\alpha{=}0.05$ nominal",
    r"empirical violation rate $\hat{\pi}$": r"taux de violation empirique $\hat{\pi}$",
    r"Violation rates: GH vs Gaussian by $\alpha$":
        r"Taux de violation : GH vs gaussienne par $\alpha$",
    "Kupiec $p$-value: Gaussian VaR": "$p$-valeur de Kupiec : VaR gaussienne",
    "Kupiec $p$-value: GH VaR": "$p$-valeur de Kupiec : VaR GH",
    r"Kupiec LR$_{\mathrm{uc}}$ $p$-values, GH vs Gaussian":
        r"$p$-valeurs de Kupiec LR$_{\mathrm{uc}}$, GH vs gaussienne",
    r"Intraday VaR backtest: $|\Delta x_n - b_x(I_n,S_n)| > \sqrt{a_{xx}(I_n,S_n)}\,q^{\,}_{1-\alpha/2}$ - GH vs Gaussian innovation, Kupiec/Christoffersen, QSE held-out sessions":
        r"Backtest de VaR intrajournalière : $|\Delta x_n - b_x(I_n,S_n)| > \sqrt{a_{xx}(I_n,S_n)}\,q^{\,}_{1-\alpha/2}$ - innovation GH vs gaussienne, Kupiec/Christoffersen, sessions QSE de test",
    # --- violation_clustering.py ------------------------------------------ #
    r"Christoffersen $p_{\mathrm{ind}}$: Flat Gaussian (no conditioning)":
        r"$p_{\mathrm{ind}}$ de Christoffersen : gaussienne plate (sans conditionnement)",
    r"Christoffersen $p_{\mathrm{ind}}$: State Gaussian $a_{xx}(I,S)$":
        r"$p_{\mathrm{ind}}$ de Christoffersen : gaussienne d'état $a_{xx}(I,S)$",
    "Book-state scale effect on violation clustering":
        "Effet de l'échelle d'état du carnet sur le regroupement des violations",
    "above diagonal = state scale\nreduces clustering":
        "au-dessus de la diagonale = l'échelle d'état\nréduit le regroupement",
    r"Christoffersen $p_{\mathrm{ind}}$: State Gaussian":
        r"$p_{\mathrm{ind}}$ de Christoffersen : gaussienne d'état",
    r"Christoffersen $p_{\mathrm{ind}}$: State GH $a_{xx}(I,S)$":
        r"$p_{\mathrm{ind}}$ de Christoffersen : GH d'état $a_{xx}(I,S)$",
    "Tail-law effect on violation clustering":
        "Effet de la loi de queue sur le regroupement des violations",
    "above diagonal = GH tail\nreduces clustering":
        "au-dessus de la diagonale = la queue GH\nréduit le regroupement",
    r"Violation clustering: Christoffersen $p_{\mathrm{ind}}$ across three VaR models (higher = less clustering evidence). Points above diagonal = improvement.":
        r"Regroupement des violations : $p_{\mathrm{ind}}$ de Christoffersen sur trois modèles de VaR (plus élevé = moins d'indices de regroupement). Points au-dessus de la diagonale = amélioration.",
    # --- tick_control.py -------------------------------------------------- #
    r"normalised diffusion $a_{xx}/\langle a_{xx}\rangle$ (within stratum)":
        r"diffusion normalisée $a_{xx}/\langle a_{xx}\rangle$ (dans la strate)",
    r"Imbalance still shapes $a_{xx}$ within fixed spread-in-ticks":
        r"Le déséquilibre façonne encore $a_{xx}$ à écart-en-pas fixé",
    r"train $\log a_{xx}$, demeaned within instrument$\times$stratum":
        r"$\log a_{xx}$ d'entraînement, centré par instrument$\times$strate",
    r"held-out $\log a_{xx}$, demeaned":
        r"$\log a_{xx}$ de test, centré",
    "Within-stratum imbalance transfer (Spearman":
        "Transfert de déséquilibre intra-strate (Spearman",
    ", weak)": ", faible)",
    r"Tick-mechanics control: imbalance still moves $a_{xx}$ within fixed spread-in-ticks (panel a), but its shape transfers only weakly out of sample (panel b)":
        r"Contrôle de la mécanique du pas : le déséquilibre déplace encore $a_{xx}$ à écart-en-pas fixé (panneau a), mais sa forme se transfère faiblement hors échantillon (panneau b)",
    # --- transfer_coldstart.py -------------------------------------------- #
    r"transferred $a_{xx}$ from other instruments  [bps$^2$]":
        r"$a_{xx}$ transféré depuis d'autres instruments  [bps$^2$]",
    r"realised held-out variance of the target  [bps$^2$]":
        r"variance de test réalisée de la cible  [bps$^2$]",
    "The surface transfers across instruments (Spearman":
        "La surface se transfère entre instruments (Spearman",
    "one-scalar floor\n(constant variance)":
        "plancher à un scalaire\n(variance constante)",
    "transferred $a_{xx}$\n(other instruments)":
        "$a_{xx}$ transféré\n(autres instruments)",
    "own $a_{xx}$\n(this instrument)":
        "$a_{xx}$ propre\n(cet instrument)",
    "OOS log-likelihood gain over the one-scalar floor":
        "gain de log-vraisemblance hors échantillon vs le plancher à un scalaire",
    "A surface from other names captures the state structure":
        "Une surface issue d'autres titres capte la structure d'état",
    "The diffusion surface is cross-sectionally portable: a shape learned from other instruments forecasts the target's state-dependent variance":
        "La surface de diffusion est portable transversalement : une forme apprise sur d'autres instruments prévoit la variance dépendant de l'état de la cible",
    # --- surface_identifiability.py --------------------------------------- #
    r"$y=x$ (perfect)": r"$y=x$ (parfait)",
    r"train scale $a_{xx}(c)$  [bps$^2$]": r"échelle d'entraînement $a_{xx}(c)$  [bps$^2$]",
    r"held-out realised variance $\mathbb{E}[(\Delta x)^2\,|\,c]$  [bps$^2$]":
        r"variance réalisée de test $\mathbb{E}[(\Delta x)^2\,|\,c]$  [bps$^2$]",
    r"The SIZE is the state: $a_{xx}$ tracks it (Spearman":
        r"La TAILLE est l'état : $a_{xx}$ la suit (Spearman",
    r"$\widehat{G}(c)=\mathbb{E}[(\Delta x)^2|c]_{\rm test}/a_{xx}^{A}(c)$":
        r"$\widehat{G}(c)=\mathbb{E}[(\Delta x)^2|c]_{\rm test}/a_{xx}^{A}(c)$",
    r"binned median $\widehat{G}$": r"médiane par classe $\widehat{G}$",
    r"$\widehat{G}=1$ (state-neutral)": r"$\widehat{G}=1$ (neutre à l'état)",
    r"independent scale $a_{xx}^{B}(c)$  [bps$^2$]  (disjoint block)":
        r"échelle indépendante $a_{xx}^{B}(c)$  [bps$^2$]  (bloc disjoint)",
    r"realised multiplier $\widehat{G}(c)$": r"multiplicateur réalisé $\widehat{G}(c)$",
    r"The leftover is state-neutral: $\widehat{{G}}\!\approx\!1$ (de-biased slope":
        r"Le résidu est neutre à l'état : $\widehat{{G}}\!\approx\!1$ (pente débiaisée",
    r"Scale vs.\ shape are separate, identified objects: $a_{xx}(I,S)$ owns the state-dependence of size; the multiplier $G$ is state-neutral":
        r"Échelle et forme sont des objets séparés, identifiés : $a_{xx}(I,S)$ porte la dépendance à l'état de la taille ; le multiplicateur $G$ est neutre à l'état",
    # --- surface_prototype.py --------------------------------------------- #
    r"state-dependent $z$": r"$z$ dépendant de l'état",
    "state-independent": "indépendant de l'état",
    r"$N(0,1)$": r"$N(0,1)$",
    "Conditioning fixes the scale, not the tails":
        "Le conditionnement fixe l'échelle, pas les queues",
    r"train $a_{xx}(c)$": r"$a_{xx}(c)$ d'entraînement",
    r"test $\mathbb{E}[(\Delta x)^2\mid c]$": r"$\mathbb{E}[(\Delta x)^2\mid c]$ de test",
    "Variance calibration (Spearman": "Calibration de la variance (Spearman",
    "state-dependent": "dépendant de l'état",
    r"cell volatility $\sqrt{a_{xx}}$ (bps)": r"volatilité de cellule $\sqrt{a_{xx}}$ (bps)",
    r"$\mathrm{std}(z)$ within cell": r"$\mathrm{std}(z)$ dans la cellule",
    "State model removes heteroskedasticity":
        "Le modèle d'état retire l'hétéroscédasticité",
    "Gaussian SDE (sim)": "EDS gaussienne (sim)",
    r"$P(|\Delta x|/\sigma_{\mathrm{loc}}>k)$": r"$P(|\Delta x|/\sigma_{\mathrm{loc}}>k)$",
    "Tails: data heavier than the model":
        "Queues : données plus lourdes que le modèle",
    r"Conditional state-dependent diffusion model vs.\ QSE data (held-out)":
        r"Modèle de diffusion conditionnel dépendant de l'état vs.\ données QSE (test)",
    # --- surface_scale_recovery.py ---------------------------------------- #
    r"$y=x$: the two measurements agree": r"$y=x$ : les deux mesures concordent",
    r"ordinary move size $\sqrt{a_{xx}}=\sqrt{\mathbb{E}[(\Delta x-b)^2]}$  (bps)":
        r"taille de mouvement ordinaire $\sqrt{a_{xx}}=\sqrt{\mathbb{E}[(\Delta x-b)^2]}$  (bps)",
    r"$(\Delta x-b)\,\div\,\mathrm{GH\ shape}\,=\,$ recovered $\sqrt{a_{xx}}$  (bps)":
        r"$(\Delta x-b)\,\div\,\mathrm{GH\ shape}\,=\,\sqrt{a_{xx}}$ récupéré  (bps)",
    r"Divide out the GH shape $\eta$ $\Rightarrow$ recover $\sqrt{{a_{{xx}}}}$ (Spearman":
        r"Diviser par la forme GH $\eta$ $\Rightarrow$ récupérer $\sqrt{{a_{{xx}}}}$ (Spearman",
    r"$(\Delta x-b)/\sqrt{G}=\sqrt{a_{xx}}\,Z$  (only $\sqrt{G}$ removed)":
        r"$(\Delta x-b)/\sqrt{G}=\sqrt{a_{xx}}\,Z$  (seul $\sqrt{G}$ retiré)",
    r"$(\Delta x-b)/\sqrt{a_{xx}G}=Z$  (both removed)":
        r"$(\Delta x-b)/\sqrt{a_{xx}G}=Z$  (les deux retirés)",
    r"$N(0,1)$ target": r"cible $N(0,1)$",
    "standardised value": "valeur standardisée",
    r"removing only $\sqrt{G}$ leaves $\sqrt{a_{xx}}\,Z$ - a scale-mixture over states $\Rightarrow$ fat tails,"
    "\n" r"not $N(0,1)$.  ($G$ is latent on real data; here $G$ is simulated, $\sqrt{a_{xx}}$ from real states.)":
        r"retirer seulement $\sqrt{G}$ laisse $\sqrt{a_{xx}}\,Z$ - un mélange d'échelles sur les états $\Rightarrow$ queues épaisses,"
        "\n" r"non $N(0,1)$.  ($G$ est latent sur données réelles ; ici $G$ est simulé, $\sqrt{a_{xx}}$ issu d'états réels.)",
    r"$\sqrt{G}$ alone can't whiten: the state scale $a_{xx}$ is still needed":
        r"$\sqrt{G}$ seul ne peut pas blanchir : l'échelle d'état $a_{xx}$ reste nécessaire",
    r"Scale and shape are both needed: $a_{xx}$ carries the state-dependence, $\sqrt{G}$ the burst --- neither alone whitens the move":
        r"Échelle et forme sont toutes deux nécessaires : $a_{xx}$ porte la dépendance à l'état, $\sqrt{G}$ la bouffée --- aucune seule ne blanchit le mouvement",
    # --- innovation_loio.py ----------------------------------------------- #
    r"borrowed $=$ own": r"empruntée $=$ propre",
    "own-fit GH gain over Gaussian (per obs)":
        "gain GH ajusté sur soi vs gaussienne (par obs)",
    "borrowed GH gain (fit on other instruments)":
        "gain GH emprunté (ajusté sur d'autres instruments)",
    "The tail law transfers across instruments":
        "La loi de queue se transfère entre instruments",
    r"NIG corner $p=-0.5$": r"coin NIG $p=-0.5$",
    r"fitted GH shape index $p$": r"indice de forme GH ajusté $p$",
    "Tail shape is nearly constant across liquidity":
        "La forme de queue est presque constante selon la liquidité",
    "The heavy-tailed innovation law is instrument-neutral: it transfers leave-one-out and its shape barely moves across liquidity":
        "La loi d'innovation à queue épaisse est neutre à l'instrument : elle se transfère en laissant-un-de-côté et sa forme bouge à peine selon la liquidité",
    # --- innovation_oos.py ------------------------------------------------ #
    "block": "bloc",
    "(empirical)": "(empirique)",
    "GH fit on": "ajustement GH sur",
    "predicting": "prédisant",
    "Gaussian fit on": "ajustement gaussien sur",
    "Fit on": "Ajusté sur",
    "predict": "prédire",
    r"$P(|z| > k)$ on the held-out block": r"$P(|z| > k)$ sur le bloc de test",
    "Out-of-sample validation of the GH tail law: fit on one temporal block, predict the disjoint block":
        "Validation hors échantillon de la loi de queue GH : ajustée sur un bloc temporel, prédit le bloc disjoint",
    # --- innovation_heavytail.py ------------------------------------------ #
    "Student-$t$ (kurtosis)": "Student-$t$ (kurtosis)",
    "NIG / GIG (kurtosis)": "NIG / GIG (kurtosis)",
    r"Panel A$'$: shape of the kick --- Gaussian vs heavy-tailed mixtures":
        r"Panneau A$'$ : forme du choc --- gaussienne vs mélanges à queue épaisse",
    r"Panel D$'$: tail exceedance --- which kick reproduces the jumps?":
        r"Panneau D$'$ : dépassement de queue --- quel choc reproduit les sauts ?",
    r"Heavy-tailed innovation $\varepsilon=\sqrt{G}\,Z$ with $Z\sim N(0,1)$ and $G$ the random variance ($\mathbb{E}[G]{=}1$): Student-$t$ (Inv.-Gamma $G$) and Gen.-Hyperbolic (GIG $G$) vs.\ Gaussian":
        r"Innovation à queue épaisse $\varepsilon=\sqrt{G}\,Z$ avec $Z\sim N(0,1)$ et $G$ la variance aléatoire ($\mathbb{E}[G]{=}1$) : Student-$t$ ($G$ inv.-gamma) et hyperbolique généralisée ($G$ GIG) vs.\ gaussienne",
    # --- innovation_simulation.py ----------------------------------------- #
    "data (held-out)": "données (test)",
    "GH-driven SDE simulation": "simulation d'EDS pilotée par GH",
    "Gaussian SDE simulation": "simulation d'EDS gaussienne",
    r"$N(0,1)$ reference": r"référence $N(0,1)$",
    r"$P(|\Delta x|/\sqrt{a_{xx}} > k)$": r"$P(|\Delta x|/\sqrt{a_{xx}} > k)$",
    "Folding the GH kick into the SDE: simulated paths now carry the jumps":
        "Intégrer le choc GH dans l'EDS : les trajectoires simulées portent désormais les sauts",
    r"tight $S$": r"$S$ étroit",
    r"mid $S$": r"$S$ moyen",
    r"wide $S$": r"$S$ large",
    r"$\mathrm{Var}(G)=\mathrm{exkurt}(z)/3$ (trimmed)":
        r"$\mathrm{Var}(G)=\mathrm{exkurt}(z)/3$ (élaguée)",
    "Tail knob by spread tercile": "Bouton de queue par tercile d'écart",
    r"order-book imbalance $I$ (bin midpoint)":
        r"déséquilibre du carnet $I$ (milieu de classe)",
    r"$\mathrm{Var}(G)$ (trimmed)": r"$\mathrm{Var}(G)$ (élaguée)",
    r"Tail knob by imbalance (Spearman$(|I|,\mathrm{{Var}}\,G)=":
        r"Bouton de queue par déséquilibre (Spearman$(|I|,\mathrm{{Var}}\,G)=",
    r"Is the fat-tail thickness $\mathrm{Var}(G)$ itself a function of the book state $(I,S)$?":
        r"L'épaisseur de queue $\mathrm{Var}(G)$ est-elle elle-même fonction de l'état du carnet $(I,S)$ ?",
}

_MISSES: set[str] = set()


def T(s):
    """Translate a user-facing label. Identity unless MICRODIFFUSION_LANG == 'fr'.

    Backward-compatible by construction: with the environment unset or 'en' this
    returns s unchanged, so English figures and the reproduction are unaffected.
    A missing French phrase falls back to English and is reported once on stderr
    so untranslated strings surface during the French regeneration.
    """
    if _LANG != "fr":
        return s
    if s in _FR:
        return _FR[s]
    if s not in _MISSES:
        _MISSES.add(s)
        print(f"[T-MISS] {s!r}", file=sys.stderr)
    return s


INK = "#1f2933"
GRID = "#d8d8d8"
ZERO = "#2f2f2f"
ACCENT = "#2f5d8c"
ACCENT_DARK = "#1d3f5f"
ACCENT_LIGHT = "#9bb6cf"
NEG = "#6f6f6f"
POS = "#2f5d8c"
WARN = "#8a5a2b"
MUTED = "#9a9a9a"
LIGHT = "#e8e8e8"

# Canonical font sizes (the intraday_reynolds.png look). Single source of truth - referenced by rcParams below.
FS_TITLE = 9.5     # axes titles
FS_LABEL = 8       # x/y axis labels
FS_TICK = 7        # tick labels
FS_LEGEND = 8      # legends
FS_TEXT = 8        # in-axes text / annotations (base font.size)
FS_SUPTITLE = 11   # figure-level suptitle
FIG_W = 13         # canonical figure WIDTH (inches) - keep uniform so on-page text size matches across figures
                   # (point sizes alone aren't enough: a narrower figure scaled to column width enlarges text)


def setup_mpl():
    """Return pyplot after applying a restrained LaTeX-like style."""
    os.environ.setdefault(
        "MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib-microdiffusion"))
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "text.usetex": False,
        "font.size": FS_TEXT,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "axes.titlesize": FS_TITLE,
        "axes.labelsize": FS_LABEL,
        "xtick.color": INK,
        "ytick.color": INK,
        "xtick.labelsize": FS_TICK,
        "ytick.labelsize": FS_TICK,
        "legend.frameon": False,
        "legend.fontsize": FS_LEGEND,
        "figure.titlesize": FS_SUPTITLE,
        "figure.dpi": 130,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.45,
        "grid.alpha": 0.65,
    })
    import matplotlib.pyplot as plt

    return plt


def despine(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return ax


def finish(fig, path):
    # tight_layout cannot solve panels with 3d axes and colorbars and warns harmlessly;
    # savefig.bbox="tight" already crops the figure, so the warning carries no information.
    import warnings
    from pathlib import Path
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Tight layout not applied")
        fig.tight_layout()
    # Write the raster the scripts have always produced, and its PDF sibling in the same
    # call so the manuscript's \includegraphics{...pdf} stays under the reproduction chain
    # (no manual out-of-chain conversion step). Both carry identical content at savefig.dpi.
    p = Path(path)
    fig.savefig(p)
    fig.savefig(p.with_suffix(".pdf"))
