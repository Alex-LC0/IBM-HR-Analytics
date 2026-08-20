# Importation de l'ensemble des packages utiles à notre analyse

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency as chi2_contingency
from scipy.stats import shapiro, pearsonr, spearmanr
from itertools import combinations
from scipy.stats import f_oneway

# Fonction de création d'histogrammes de distribution :


def distribution(dataframe: list, s: int, t: int):
    """
    Affiche les histogrammes de distribution pour chaque variable (colonne) d'un jeu de données.

    Pour chaque clé (colonne) du jeu de données fourni, la fonction trace un histogramme
    représentant la distribution des valeurs associées, puis affiche l'ensemble des
    histogrammes sous forme de sous-graphiques (subplots) organisés sur une grille
    de n lignes et 3 colonnes.

    Parameters
    ----------
    dataframe : list ou pandas.DataFrame
        Jeu de données dont les colonnes/clés (accessibles via .keys()) seront utilisées
        pour tracer les histogrammes. Doit posséder une méthode .keys() ainsi qu'un
        accès indexé par clé (dataframe[k]).
    s : int
        Largeur de la figure matplotlib (en pouces), utilisée dans figsize=(s, t).
    t : int
        Hauteur de la figure matplotlib (en pouces), utilisée dans figsize=(s, t).

    Returns
    -------
    None
        La fonction ne retourne rien : elle affiche directement la figure générée
        via plt.show().

    Notes
    -----
    Le nombre de lignes de la grille de subplots est déterminé par n = len(dataframe).
    Attention : si `dataframe` est un pandas.DataFrame, len(dataframe) correspond au
    nombre de lignes (observations) et non au nombre de colonnes (variables), ce qui
    peut entraîner une grille de subplots mal dimensionnée par rapport au nombre
    réel de clés à afficher.

    Examples
    --------
    >>> distribution(df, 15, 10)
    """
    n = len(dataframe)
    plt.figure(figsize=(s, t))

    for k, i in zip(dataframe.keys(), range(1, n + 1)):
        plt.subplot(n, 3, i)
        plt.hist(dataframe[k])
        plt.title(k)
    plt.show()

# Fonction de Segmentation de l'age :


def segmentation_age(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Détermine la tranche d'âge correspondant à une valeur d'âge donnée.

    Cette fonction segmente l'âge en 4 catégories basées sur les quartiles de la
    distribution (Q1, Q2/médiane, Q3), et retourne le libellé de la tranche
    correspondante. Elle est destinée à être appliquée ligne par ligne à un
    DataFrame (par exemple via dataframe.apply(segmentation_age, axis=1)).

    Parameters
    ----------
    dataframe : pd.DataFrame
        En pratique, une ligne (pd.Series) issue d'un DataFrame, possédant une
        clé 'Age' (dataframe['Age']) contenant une valeur numérique d'âge.

    Returns
    -------
    str
        Libellé de la tranche d'âge correspondante, parmi :
        - '18 - 30 ans' si Age < 30 (du minimum au 1er quartile)
        - '31 - 36 ans' si 30 <= Age < 36 (du 1er au 2e quartile)
        - '37 - 43 ans' si 36 <= Age < 43 (du 3e au 4e quartile)
        - '44 - 60 ans' si Age >= 43 (du 4e quartile au maximum)

    Notes
    -----
    - Le type de retour annoncé dans la signature (pd.DataFrame) ne correspond pas
      au type réellement retourné (str).
    - Les seuils (30, 36, 43) sont codés en dur et correspondent aux quartiles
      observés sur un jeu de données spécifique ; ils ne sont pas recalculés
      dynamiquement à partir des données passées en paramètre.
    - Cette fonction attend un objet indexable par 'Age' (typiquement une ligne de
      DataFrame) et non un DataFrame entier, malgré le nom du paramètre.

    Examples
    --------
    >>> dataframe['Tranche_Age'] = dataframe.apply(segmentation_age, axis=1)
    """
    if dataframe["Age"] < 30:  # de min à q1
        return "18 - 30 ans"
    elif dataframe["Age"] < 36:  # de q1 à q2
        return "31 - 36 ans"
    elif dataframe["Age"] < 43:  # de q3 à q4
        return "37 - 43 ans"
    else:  # de q4 à max
        return "44 - 60 ans"


# Fonction de segmentation de YearsAtCompagny


def segmentation_yac(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Détermine la tranche d'ancienneté (en années dans l'entreprise) correspondant à une valeur donnée.

    Cette fonction segmente l'ancienneté (YearsAtCompany) en 4 catégories, et retourne
    le libellé de la tranche correspondante. Elle est destinée à être appliquée ligne
    par ligne à un DataFrame (par exemple via dataframe.apply(segmentation_yac, axis=1)).

    Parameters
    ----------
    dataframe : pd.DataFrame
        En pratique, une ligne (pd.Series) issue d'un DataFrame, possédant une
        clé 'YearsAtCompany' (dataframe['YearsAtCompany']) contenant une valeur
        numérique d'ancienneté en années.

    Returns
    -------
    str
        Libellé de la tranche d'ancienneté correspondante, parmi :
        - '0 à 5 ans' si YearsAtCompany <= 5
        - '6 à 10 ans' si 5 < YearsAtCompany <= 10
        - '11 à 20 ans' si 10 < YearsAtCompany <= 20
        - '+ de 20 ans' si YearsAtCompany > 20

    Notes
    -----
    - Le type de retour annoncé dans la signature (pd.DataFrame) ne correspond pas
      au type réellement retourné (str).
    - Les seuils (5, 10, 20) sont codés en dur dans la fonction.
    - Cette fonction attend un objet indexable par 'YearsAtCompany' (typiquement une
      ligne de DataFrame) et non un DataFrame entier, malgré le nom du paramètre.

    Examples
    --------
    >>> dataframe['Tranche_YAC'] = dataframe.apply(segmentation_yac, axis=1)
    """
    if dataframe["YearsAtCompany"] <= 5:
        return "0 à 5 ans"
    elif dataframe["YearsAtCompany"] <= 10:
        return "6 à 10 ans"
    elif dataframe["YearsAtCompany"] <= 20:
        return "11 à 20 ans"
    else:
        return "+ de 20 ans"


# Fonction de segmentation de YearsInCurrentRole


def segmentation_yar(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Détermine la tranche d'ancienneté dans le poste actuel correspondant à une valeur donnée.

    Cette fonction segmente l'ancienneté dans le rôle actuel (YearsInCurrentRole) en
    4 catégories, et retourne le libellé de la tranche correspondante. Elle est
    destinée à être appliquée ligne par ligne à un DataFrame (par exemple via
    dataframe.apply(segmentation_yar, axis=1)).

    Parameters
    ----------
    dataframe : pd.DataFrame
        En pratique, une ligne (pd.Series) issue d'un DataFrame, possédant une
        clé 'YearsInCurrentRole' (dataframe['YearsInCurrentRole']) contenant une
        valeur numérique d'ancienneté en années dans le poste actuel.

    Returns
    -------
    str
        Libellé de la tranche d'ancienneté correspondante, parmi :
        - '0 à 3 ans' si YearsInCurrentRole <= 3
        - '4 à 7 ans' si 3 < YearsInCurrentRole <= 7
        - '8 à 15 ans' si 7 < YearsInCurrentRole <= 15
        - '+ de 15 ans' si YearsInCurrentRole > 15

    Notes
    -----
    - Le type de retour annoncé dans la signature (pd.DataFrame) ne correspond pas
      au type réellement retourné (str).
    - Les seuils (3, 7, 15) sont codés en dur dans la fonction.
    - Cette fonction attend un objet indexable par 'YearsInCurrentRole' (typiquement
      une ligne de DataFrame) et non un DataFrame entier, malgré le nom du paramètre.

    Examples
    --------
    >>> dataframe['Tranche_YIR'] = dataframe.apply(segmentation_yar, axis=1)
    """
    if dataframe["YearsInCurrentRole"] <= 3:
        return "0 à 3 ans"
    elif dataframe["YearsInCurrentRole"] <= 7:
        return "4 à 7 ans"
    elif dataframe["YearsInCurrentRole"] <= 15:
        return "8 à 15 ans"
    else:
        return "+ de 15 ans"


# Fonction de transformation de l'attrition en 'int'


def attrition_int(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Convertit la variable catégorielle 'Attrition' en variable binaire numérique.

    Cette fonction transforme la valeur textuelle de la colonne 'Attrition'
    ('Yes'/'No') en une valeur entière (1/0), afin de faciliter les traitements
    statistiques ou les calculs de corrélation. Elle est destinée à être appliquée
    ligne par ligne à un DataFrame (par exemple via dataframe.apply(attrition_int, axis=1)).

    Parameters
    ----------
    dataframe : pd.DataFrame
        En pratique, une ligne (pd.Series) issue d'un DataFrame, possédant une
        clé 'Attrition' (dataframe['Attrition']) contenant la valeur 'Yes' ou 'No'.

    Returns
    -------
    int
        1 si la valeur de 'Attrition' est 'Yes' (départ de l'employé),
        0 sinon (y compris pour toute autre valeur que 'Yes').

    Notes
    -----
    - Le type de retour annoncé dans la signature (pd.DataFrame) ne correspond pas
      au type réellement retourné (int).
    - Cette fonction attend un objet indexable par 'Attrition' (typiquement une
      ligne de DataFrame) et non un DataFrame entier, malgré le nom du paramètre.
    - La comparaison est sensible à la casse et à l'orthographe exacte ('Yes').

    Examples
    --------
    >>> dataframe['Attrition_bin'] = dataframe.apply(attrition_int, axis=1)
    """
    if dataframe["Attrition"] == "Yes":
        return 1
    else:
        return 0


# Fonction de corrélation entre les variables :

QUANTITATIVE = [
    "Age",
    "DailyRate",
    "DistanceFromHome",
    "HourlyRate",
    "MonthlyIncome",
    "YearsAtCompany",
    "YearsInCurrentRole",
    "YearsSinceLastPromotion",
    "YearsWithCurrManager",
]
QUALITATIVE_NOM = ["Department"]
QUALITATIVE_ORD = [
    "Attrition",
    "Education",
    "EnvironmentSatisfaction",
    "JobInvolvement",
    "JobSatisfaction",
    "OverTime",
]


def an_corr(dataframe: pd.DataFrame) -> str:
    """
    Analyse et affiche les corrélations entre paires de variables, en adaptant le test
    statistique utilisé selon le type des variables (quantitatif, qualitatif nominal
    ou qualitatif ordinal).

    Pour chaque paire distincte de colonnes du DataFrame (obtenue via combinations),
    la fonction applique le test le plus adapté :
    - deux variables quantitatives (QUANTITATIVE) : test de normalité de Shapiro-Wilk
      sur chacune des deux variables ; si les deux distributions sont considérées
      comme normales (p-value > alpha), corrélation de Pearson (np.corrcoef) ;
      sinon, corrélation de Spearman (spearmanr) ;
    - deux variables qualitatives ordinales (QUALITATIVE_ORD), ou une variable
      qualitative ordinale et une variable qualitative nominale (QUALITATIVE_NOM) :
      test du Chi² d'indépendance (chi2_contingency) sur un tableau de contingence
      (pd.crosstab) ;
    - une variable quantitative (QUANTITATIVE) et la variable nominale 'Department'
      (QUALITATIVE_NOM) : analyse de variance à un facteur (ANOVA, f_oneway) entre
      les trois modalités 'Sales', 'Research & Development' et 'Human Resources' ;
    - toute autre combinaison de types n'est pas traitée (aucune branche associée).

    Dans tous les cas, le résultat du test (coefficient de corrélation, p-value,
    statistique du test, etc.) est affiché via print().

    Parameters
    ----------
    dataframe : pd.DataFrame
        Jeu de données contenant les colonnes à analyser. Les colonnes concernées
        par les tests statistiques doivent figurer dans les listes globales
        QUANTITATIVE, QUALITATIVE_NOM ou QUALITATIVE_ORD pour être traitées.
        Pour le test ANOVA, la colonne 'Department' (QUALITATIVE_NOM) doit contenir
        au moins les modalités 'Sales', 'Research & Development' et
        'Human Resources'.

    Returns
    -------
    None
        La fonction ne retourne rien : les résultats sont affichés directement via
        print() et non renvoyés, malgré le type de retour annoncé (str).

    Notes
    -----
    - Le type de retour annoncé dans la signature (str) ne correspond pas au
      comportement réel de la fonction, qui ne retourne rien (None).
    - Les listes QUANTITATIVE, QUALITATIVE_NOM et QUALITATIVE_ORD sont des variables
      globales définies en dehors de la fonction, et codées en dur (indépendantes
      des colonnes réellement présentes dans `dataframe`).
    - Le seuil `alpha = 0.5` utilisé pour juger de la normalité (test de Shapiro-Wilk)
      est plus permissif que le seuil conventionnel de 0.05, ce qui augmente le
      risque de choisir Pearson pour des distributions non normales.
    - Le test ANOVA (f_oneway) suppose que les modalités 'Sales',
      'Research & Development' et 'Human Resources' sont présentes dans la colonne
      'Department' ; toute autre modalité est ignorée, et l'absence d'une de ces
      trois modalités provoquera une erreur (groupe vide).
    - Les combinaisons quantitatif/qualitatif ordinal, ainsi que nominal/nominal, ne
      sont pas traitées.
    - Nécessite l'import de `combinations` (itertools), `shapiro`, `spearmanr`,
      `chi2_contingency` et `f_oneway` (scipy.stats).

    Examples
    --------
    >>> an_corr(df)
    La corrélation entre Age et DailyRate est de 0.0103 (pearson)
    La corrélation de Attrition et Education est de 0.452 (chi2 = 1.85, dof = 4)
    La corrélation entre MonthlyIncome et Department est de 0.0021 (fisher = 6.42)
    """
    for i, k in combinations(dataframe.columns, 2):
        if i in QUANTITATIVE and k in QUANTITATIVE:
            alpha = 0.5
            p_val_i = shapiro(dataframe[i]).pvalue
            p_val_k = shapiro(dataframe[k]).pvalue
            if p_val_i > alpha and p_val_k > alpha:
                pearson_cor = np.corrcoef(dataframe[i], dataframe[k])
                print(
                    f"La corrélation entre {i} et {k} est de {pearson_cor[0,1]} (pearson)"
                )
            else:
                spearman_corr = spearmanr(dataframe[i], dataframe[k])
                print(
                    f"La corrélation entre {i} et {k} est de {spearman_corr.correlation} (spearman)"
                )
        elif (
            i in QUALITATIVE_ORD
            and k in QUALITATIVE_ORD
            or i in QUALITATIVE_ORD
            and k in QUALITATIVE_NOM
        ):
            table = pd.crosstab(dataframe[i], dataframe[k])
            chi2, p_value, dof, expected = chi2_contingency(table)
            print(
                f"La corrélation de {i} et {k} est de {p_value} (chi2 = {chi2}, dof = {dof})"
            )
        elif i in QUANTITATIVE and k in QUALITATIVE_NOM:
            group1 = dataframe[dataframe[k] == "Sales"][i]
            group2 = dataframe[dataframe[k] == "Research & Development"][i]
            group3 = dataframe[dataframe[k] == "Human Resources"][i]
            f_stat, p_val = f_oneway(group1, group2, group3)
            print(f"La corrélation entre {i} et {k} est de {p_val} (fisher = {f_stat})")


# Fonction de création de Plot Graph de comparaison :


def plot_bar(average: pd.DataFrame, titre: str, arrondi: int) -> plt:
    """
    Trace un diagramme en barres comparant une moyenne (ou valeur agrégée) entre
    employés restés dans l'entreprise et employés partis (attrition).

    La fonction affiche deux barres, l'une représentant la valeur associée aux
    employés restés ('No'), l'autre aux employés partis ('Yes'), avec les valeurs
    numériques annotées au-dessus de chaque barre.

    Parameters
    ----------
    average : pd.DataFrame
        Structure indexable par 'No' et 'Yes' (par ex. une pd.Series résultant d'un
        groupby sur la variable 'Attrition'), contenant les valeurs moyennes (ou
        agrégées) à comparer pour chacune des deux modalités.
    titre : str
        Titre de la variable analysée, utilisé pour composer le titre du graphique
        sous la forme "{titre} v/s attrition".
    arrondi : int
        Nombre de décimales à afficher pour les valeurs annotées au-dessus des
        barres (format f"{value:,.{arrondi}f}").

    Returns
    -------
    None
        La fonction ne retourne rien : elle construit et affiche la figure
        matplotlib directement (via plt.figure, plt.bar, etc.), malgré le type de
        retour annoncé (plt).

    Notes
    -----
    - Le type de retour annoncé dans la signature (plt) n'est pas un type valide
      et ne correspond pas au comportement réel de la fonction, qui ne retourne
      rien (None).
    - La fonction ne comporte pas d'appel à plt.show() : l'affichage effectif du
      graphique dépendra du contexte d'exécution (par ex. mode interactif, notebook).
    - Les couleurs des barres sont fixées en dur ('Green' pour 'No', 'Red' pour
      'Yes'), et les labels de l'axe des x supposent implicitement que 'average'
      est bien structuré selon l'ordre ['No', 'Yes'].
    - La fonction suppose que `average` possède exactement les clés 'No' et 'Yes'
      (typiquement issues de la variable 'Attrition') ; toute autre structure
      provoquera une erreur.

    Examples
    --------
    >>> moyenne_revenu = dataframe.groupby('Attrition')['MonthlyIncome'].mean()
    >>> plot_bar(moyenne_revenu, "Revenu mensuel moyen", 2)
    """
    plt.figure(figsize=(7, 5))

    labels = ["Reste(No)", "Parti(Yes)"]

    average_value = [average["No"], average["Yes"]]

    propriete = dict(color=["Green", "Red"], edgecolor="Black")

    bar = plt.bar(labels, average_value, **propriete)
    plt.title(f"{titre} v/s attrition")

    for i, value in enumerate(average_value):
        plt.text(
            i,
            value,
            f"{value:,.{arrondi}f}",
            ha="center",
            fontsize=12,
            fontweight="bold",
        )
    plt.tight_layout()


# Creation de HistBar


def plot_hist(
    average1: pd.DataFrame, average2: pd.DataFrame, titre: str, xlb: str, ylb: str
) -> plt:
    """
    Trace deux histogrammes superposés comparant la distribution d'une variable
    entre employés partis et employés restés (attrition).

    La fonction affiche sur un même graphique l'histogramme de la variable pour les
    employés partis ('Parti') en opacité forte, et celui des employés restés
    ('Reste') en opacité faible, afin de visualiser la différence de distribution
    entre les deux groupes.

    Parameters
    ----------
    average1 : pd.DataFrame
        Série de valeurs (par ex. une colonne filtrée sur Attrition == 'Yes')
        représentant la distribution de la variable pour les employés partis.
        Affiché en légende sous le libellé 'Parti'.
    average2 : pd.DataFrame
        Série de valeurs (par ex. une colonne filtrée sur Attrition == 'No')
        représentant la distribution de la variable pour les employés restés.
        Affiché en légende sous le libellé 'Reste'.
    titre : str
        Titre de la variable analysée, utilisé pour composer le titre du graphique
        sous la forme "{titre} v/s Attrition".
    xlb : str
        Libellé de l'axe des abscisses (xlabel).
    ylb : str
        Libellé de l'axe des ordonnées (ylabel).

    Returns
    -------
    None
        La fonction ne retourne rien : elle construit la figure matplotlib
        directement (via plt.figure, plt.hist, etc.), malgré le type de retour
        annoncé (plt).

    Notes
    -----
    - Le type de retour annoncé dans la signature (plt) n'est pas un type valide
      et ne correspond pas au comportement réel de la fonction, qui ne retourne
      rien (None).
    - La fonction ne comporte pas d'appel à plt.show() : l'affichage effectif du
      graphique dépendra du contexte d'exécution (par ex. mode interactif, notebook).
    - L'association des libellés 'Parti'/'Reste' à average1/average2 est faite par
      convention (average1 = partis, average2 = restés) et n'est pas vérifiée par
      la fonction : il appartient à l'appelant de fournir les données dans le bon
      ordre.
    - Les niveaux de transparence (alpha=0.9 pour average1, alpha=0.2 pour
      average2) sont fixés en dur.

    Examples
    --------
    >>> partis = dataframe[dataframe['Attrition'] == 'Yes']['MonthlyIncome']
    >>> restes = dataframe[dataframe['Attrition'] == 'No']['MonthlyIncome']
    >>> plot_hist(partis, restes, "Revenu mensuel", "Revenu ($)", "Effectif")
    """
    plt.figure(figsize=(8, 6))
    plt.hist(average1, label="Parti", alpha=0.9, edgecolor="Black")
    plt.hist(average2, label="Reste", alpha=0.2, edgecolor="Black")
    plt.xlabel(xlb)
    plt.ylabel(ylb)
    plt.legend()
    plt.title(f"{titre} v/s Attrition")


# Fonction de corrélation {à retirer du jupyt car sert à rien}


def correlation(dataframe: pd.DataFrame) -> list:
    """
    Calcule les coefficients de corrélation de Pearson entre plusieurs variables numériques.

    Pour chaque paire distincte de variables parmi une liste prédéfinie de colonnes
    (Age, DailyRate, DistanceFromHome, HourlyRate, YearsAtCompany, YearsInCurrentRole,
    YearsSinceLastPromotion, YearsWithCurrManager), la fonction calcule le coefficient
    de corrélation de Pearson à l'aide de np.corrcoef.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Jeu de données contenant obligatoirement les colonnes suivantes : 'Age',
        'DailyRate', 'DistanceFromHome', 'HourlyRate', 'YearsAtCompany',
        'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager'.

    Returns
    -------
    dict
        Dictionnaire dont les clés sont des tuples (x, y) représentant les noms des
        deux variables comparées, et dont les valeurs sont les coefficients de
        corrélation de Pearson (float, compris entre -1 et 1) associés.

    Notes
    -----
    - La signature indique un type de retour `list`, mais la fonction retourne en
      réalité un `dict`.
    - Chaque paire de variables distinctes (x, y) est calculée deux fois (une fois
      en (x, y) et une fois en (y, x)), les résultats étant symétriques puisque
      corr(x, y) == corr(y, x).
    - Les paires où x == y (auto-corrélation, toujours égale à 1) sont exclues.
    - La liste des variables analysées (CORRELATION_x et CORRELATION_y) est fixée en
      dur dans la fonction et ne dépend pas des colonnes réellement présentes dans
      `dataframe`, ce qui provoquera une erreur si l'une de ces colonnes est absente.

    Examples
    --------
    >>> resultats = correlation(df)
    >>> resultats[('Age', 'DailyRate')]
    0.0103
    """
    CORRELATION_x = [
        "Age",
        "DailyRate",
        "DistanceFromHome",
        "HourlyRate",
        "YearsAtCompany",
        "YearsInCurrentRole",
        "YearsSinceLastPromotion",
        "YearsWithCurrManager",
    ]
    CORRELATION_y = [
        "Age",
        "DailyRate",
        "DistanceFromHome",
        "HourlyRate",
        "YearsAtCompany",
        "YearsInCurrentRole",
        "YearsSinceLastPromotion",
        "YearsWithCurrManager",
    ]
    resultat_corr = {}
    for x in CORRELATION_x:
        for y in CORRELATION_y:
            if x != y:
                resultat_corr[(x, y)] = np.corrcoef(dataframe[x], dataframe[y])[0, 1]
    return resultat_corr


def plot_analyse(attrition_rate : int | float, titre : str, xlabel : str) : 
    """
    Trace un diagramme en barres représentant un taux d'attrition par modalité.

    La fonction affiche une barre par modalité (index) de `attrition_rate`, avec un
    jeu de couleurs fixe se répétant par paires (Bleu, Bleu, Vert, Vert, Rouge,
    Rouge), pensé pour un axe comportant exactement 6 modalités.

    Parameters
    ----------
    attrition_rate : int | float
        En pratique, une structure de type pd.Series (et non int | float),
        possédant un attribut `.index` (les modalités affichées en abscisse) et
        `.values` (les taux d'attrition associés, affichés en ordonnée). Typiquement
        le résultat d'un groupby suivi d'une agrégation (ex. moyenne d'attrition
        par modalité).
    titre : str
        Titre du graphique (plt.title).
    xlabel : str
        Libellé de l'axe des abscisses ; également réutilisé pour composer le
        libellé de l'axe des ordonnées sous la forme "{xlabel} attrition rate".

    Returns
    -------
    None
        La fonction ne retourne rien : elle construit la figure matplotlib
        directement (via plt.figure, plt.bar, etc.).

    Notes
    -----
    - Le type annoncé pour `attrition_rate` (int | float) ne correspond pas au type
      réellement attendu (un objet possédant .index et .values, typiquement une
      pd.Series), et ne figure pas dans la signature d'origine sous forme de
      docstring cohérente avec l'usage réel.
    - La liste de couleurs `['Blue','Blue','Green','Green', 'Red', 'Red']` est fixée
      en dur et suppose exactement 6 modalités dans `attrition_rate.index` ; un
      nombre différent de modalités entraînera une erreur (trop de couleurs) ou une
      réutilisation partielle/incohérente des couleurs selon le comportement de
      matplotlib.
    - `plt.tight_layout` est appelé sans parenthèses : il s'agit d'une référence à
      la fonction et non d'un appel de fonction, ce qui signifie que l'ajustement
      de la mise en page n'est en réalité jamais effectué.
    - La fonction ne comporte pas d'appel à plt.show() : l'affichage effectif du
      graphique dépendra du contexte d'exécution (par ex. mode interactif, notebook).

    Examples
    --------
    >>> taux = dataframe.groupby('BusinessTravel')['Attrition_bin'].mean()
    >>> plot_analyse(taux, "Attrition selon les déplacements professionnels", "BusinessTravel")
    """
    plt.figure(figsize=(8,6))
    plt.bar(attrition_rate.index, attrition_rate.values, color=['Blue','Blue','Green','Green', 'Red', 'Red'], edgecolor='Black', alpha=0.5)
    plt.title(titre)
    plt.xlabel(xlabel)
    plt.ylabel(f"{xlabel} attrition rate")
    plt.tight_layout()

def segmentation_promo(dataframe : pd.DataFrame) : 
    """
    Détermine la tranche d'années depuis la dernière promotion correspondant à une valeur donnée.

    Cette fonction segmente le nombre d'années depuis la dernière promotion
    (YearsSinceLastPromotion) en 3 catégories, et retourne le libellé de la tranche
    correspondante. Elle est destinée à être appliquée ligne par ligne à un
    DataFrame (par exemple via dataframe.apply(segmentation_promo, axis=1)).

    Parameters
    ----------
    dataframe : pd.DataFrame
        En pratique, une ligne (pd.Series) issue d'un DataFrame, possédant une
        clé 'YearsSinceLastPromotion' (dataframe['YearsSinceLastPromotion'])
        contenant une valeur numérique d'années depuis la dernière promotion.

    Returns
    -------
    str
        Libellé de la tranche correspondante, parmi :
        - '0 à 4 ans' si YearsSinceLastPromotion < 4
        - '5 - 8 ans' si 4 <= YearsSinceLastPromotion < 8
        - '9 - 14 ans' si YearsSinceLastPromotion >= 8

    Notes
    -----
    - Les seuils (4, 8) sont codés en dur dans la fonction.
    - Le libellé de la dernière tranche ('9 - 14 ans') suppose une valeur maximale
      de 14 ans, information non vérifiée par la fonction : toute valeur >= 8 sera
      classée dans cette tranche, même si elle dépasse 14.
    - Cette fonction attend un objet indexable par 'YearsSinceLastPromotion'
      (typiquement une ligne de DataFrame) et non un DataFrame entier, malgré le
      nom du paramètre.
    - La signature ne précise pas de type de retour explicite.

    Examples
    --------
    >>> dataframe['Tranche_Promo'] = dataframe.apply(segmentation_promo, axis=1)
    """
    if dataframe["YearsSinceLastPromotion"] < 4 : 
        return ("0 à 4 ans")
    elif dataframe['YearsSinceLastPromotion'] < 8 : 
        return("5 - 8 ans")
    else :
        return("9 - 14 ans")

def segmentation_ywcm(dataframe : pd.DataFrame) : 
    """
    Détermine la tranche d'ancienneté avec le manager actuel correspondant à une valeur donnée.

    Cette fonction segmente l'ancienneté avec le manager actuel (YearsWithCurrManager)
    en 4 catégories, et retourne le libellé de la tranche correspondante. Elle est
    destinée à être appliquée ligne par ligne à un DataFrame (par exemple via
    dataframe.apply(segmentation_ywcm, axis=1)).

    Parameters
    ----------
    dataframe : pd.DataFrame
        En pratique, une ligne (pd.Series) issue d'un DataFrame, possédant une
        clé 'YearsWithCurrManager' (dataframe['YearsWithCurrManager']) contenant
        une valeur numérique d'ancienneté en années avec le manager actuel.

    Returns
    -------
    str
        Libellé de la tranche d'ancienneté correspondante, parmi :
        - '0 à 2 ans' si YearsWithCurrManager <= 2
        - '3 à 5 ans' si 2 < YearsWithCurrManager <= 5
        - '6 à 8 ans' si 5 < YearsWithCurrManager <= 8
        - 'Plus de 8 ans' si YearsWithCurrManager > 8

    Notes
    -----
    - Les seuils (2, 5, 8) sont codés en dur dans la fonction.
    - Cette fonction attend un objet indexable par 'YearsWithCurrManager'
      (typiquement une ligne de DataFrame) et non un DataFrame entier, malgré le
      nom du paramètre.

    Examples
    --------
    >>> dataframe['Tranche_YWCM'] = dataframe.apply(segmentation_ywcm, axis=1)
    """
    if dataframe['YearsWithCurrManager'] <= 2 :
        return ("0 à 2 ans")
    elif dataframe['YearsWithCurrManager'] <= 5 : 
        return("3 à 5 ans")
    elif dataframe['YearsWithCurrManager'] <=8 : 
        return("6 à 8 ans")
    else : 
        return("Plus de 8 ans")

def plot_details(data : int | float, titre : str, xlabel : str, ylabel : str) : 
    """
    Trace un diagramme en barres à partir de valeurs moyennes, pour une variable
    comportant jusqu'à 3 modalités, avec annotation de la valeur au-dessus de
    chaque barre.

    La fonction affiche un histogramme en barres à partir des index (modalités) et
    des valeurs moyennes de `data`, avec un jeu de couleurs fixe (Bleu, Orange,
    Vert), et annote chaque barre avec sa valeur numérique arrondie à 1 décimale.

    Parameters
    ----------
    data : pd.Series
        Série indexée par les modalités d'une variable catégorielle et contenant
        les valeurs moyennes associées à chaque modalité (typiquement issue d'un
        groupby suivi d'une agrégation .mean()). L'index doit comporter au maximum
        3 modalités pour correspondre au jeu de couleurs défini.
    titre : str
        Titre du graphique.
    xlabel : str
        Libellé de l'axe des abscisses.
    ylabel : str
        Libellé de l'axe des ordonnées.

    Returns
    -------
    None
        La fonction ne retourne rien : elle construit et affiche la figure
        matplotlib directement (via plt.bar, plt.text, etc.).

    Notes
    -----
    - Le type annoncé pour `data` (int | float) ne correspond pas à l'usage réel
      dans la fonction, qui accède à `.index` et `.values` : il s'agit donc en
      pratique d'une pd.Series de moyennes (issue par exemple d'un
      groupby(...).mean()), et non d'un int ou d'un float.
    - La liste de couleurs `['Blue', 'Orange', 'Green']` est fixée en dur et
      suppose que `data` comporte au maximum 3 modalités ; un nombre de modalités
      supérieur à 3 provoquera une erreur (couleurs manquantes).
    - Le nombre de décimales affichées dans les annotations (1) est fixé en dur
      (f"{value:,.1f}"), contrairement à d'autres fonctions similaires du module
      (ex. plot_bar) où ce paramètre est configurable.
    - La fonction ne comporte pas d'appel à plt.figure() : elle s'appuie sur une
      figure déjà existante ou créée par défaut par matplotlib, ni d'appel à
      plt.show(), l'affichage effectif dépendant du contexte d'exécution.

    Examples
    --------
    >>> moyenne = dataframe.groupby('MaritalStatus')['MonthlyIncome'].mean()
    >>> plot_details(moyenne, "Revenu mensuel moyen par statut marital", "Statut marital", "Revenu moyen ($)")
    """
    plt.bar(data.index, data.values, color=['Blue', 'Orange', 'Green'], edgecolor='Black')
    plt.title(titre)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    for i, value in enumerate(data):
        plt.text(
            i,
            value,
            f"{value:,.1f}",
            ha="center",
            fontsize=12,
            fontweight="bold",
        )
    plt.tight_layout()