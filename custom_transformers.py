"""
Contains custom transformer classes and functions to be imported into preparation and training notebooks.

Main entry point:  make_preprocessor(...)  builds the full preprocessing pipeline.
It takes a raw sightings table (the train or test set, with or without the `diseased` column)
and returns a table of numbers that is ready to train on.
"""

import re

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler
from sklearn.utils.validation import check_is_fitted

# The 13 behavior tick boxes of the census (True / False)
BEHAVIORS = ["Running", "Chasing", "Climbing", "Eating", "Foraging", "Kuks", "Quaas", "Moans",
             "Tail flags", "Tail twitches", "Approaches", "Indifferent", "Runs from"]

# Columns that must be in every table the preprocessor receives
REQUIRED_COLUMNS = ["X", "Y", "Shift", "Age", "Primary Fur Color", "Highlight Fur Color", "Location",
                    "Above Ground Sighter Measurement"] + BEHAVIORS

# Day-level columns (one value per day): only used when keep_day_level=True
DAY_LEVEL_NUMERIC = ["wx_tmax_c", "wx_tmin_c", "wx_prcp_mm", "wx_prcp_prev_day_mm", "wx_wind_ms",
                     "cal_day_of_week", "cal_days_to_holiday"]
DAY_LEVEL_BINARY = ["cal_is_weekend"]


# Free-text columns where a volunteer may have written what they noticed about the animal.
# `Specific Location` is left out on purpose: it describes a place, not the squirrel.
TEXT_COLUMNS = ["Color notes", "Other Activities", "Other Interactions"]

# Keywords by theme. They come from the symptoms in the Frame step (aggression, foaming at the mouth, mangy fur
# "as if the squirrel had been attacked", a rabies-like illness) and from plain-language synonyms. They were NOT picked by
# looking at which words go with the label. Left out on purpose:
#   "mad"   the census notes contain "made executive adjustments" in 40+ rows, so it would match that text
#   "mouth" the notes say "nut in mouth" or "food in mouth", which is normal eating
#   "chase" / "chasing" is already a tick box, and most chases are play or a dog chasing the squirrel
#   "patchy" coat color notes use "patch" ("white patch on the back"), so it would match normal fur
KEYWORDS = {
    "aggression": ["aggressive", "aggression", "attack", "attacking", "attacked", "angry", "furious", "rabid", "hostile",
                   "fight", "fighting", "bite", "bites", "biting", "bitten", "snarl", "snarling", "growl", "growling",
                   "hiss", "hissing", "lunge", "lunging", "charging", "threatening"],
    "foaming":    ["foam", "foaming", "foamy", "froth", "frothing", "drool", "drooling", "saliva", "slobber"],
    "fur":        ["mangy", "mange", "bald", "balding", "hairless", "scab", "scabs", "scabby", "wound",
                   "wounded", "injured", "injury", "bleeding", "scratched", "scratching", "itchy", "sore", "sores"],
    "ill":        ["sick", "unwell", "lethargic", "weak", "stumbling", "staggering", "limp", "limping", "disoriented",
                   "confused", "erratic", "strange", "unafraid", "fearless", "trembling", "twitching", "seizure",
                   "unsteady"],
}

# Text the census app adds by itself ("gray & cinnamon selected as primary. ... made executive adjustments.")
_BOILERPLATE = [re.compile(r"[^.]*\bselected as (?:primary|highlights?)\b[^.]*\.?"),
                re.compile(r"made executive adjustments\.?")]


_SUFFIXES = ("ing", "ed", "ly", "ive", "ion", "ness", "es", "s", "y", "e")


def stem(word):
    """A very small stemmer: 'aggressive', 'aggression' and 'aggressively' all become 'aggress'.
    A word is never cut shorter than 4 letters, so short words ('bit', 'bite') are left alone."""
    changed = True
    while changed:
        changed = False
        for suffix in _SUFFIXES:
            if word.endswith(suffix) and len(word) - len(suffix) >= 4 and not (suffix == "s" and word.endswith("ss")):
                word, changed = word[: -len(suffix)], True
                break
    return word


def stems(text):
    """The stems of the words (3 letters or more) of a text. Used as the 'analyzer' of the TF-IDF vectorizer."""
    return [stem(w) for w in re.findall(r"[a-z]{3,}", text)]


class KeywordSimilarity(BaseEstimator, TransformerMixin):
    """How close is each note to the symptom keywords? (cosine similarity, one column per theme)

    1. The note is cleaned (lower case, census boilerplate removed) and cut into word stems, so "aggressively" matches
       "aggressive" and "foaming" matches "foam", but "curious" does not match "furious" and "watching" does not match
       "scratching" (they share letters, not stems).
    2. Notes and keywords become TF-IDF vectors of stems (rare stems count more). The IDF weights are learned in `fit` from
       the training notes plus the keyword list. The label is never used.
    3. For each theme the cosine similarity between the note and every keyword of the theme is computed and the
       **largest** one is kept: 0 = nothing in common, 1 = the note is that keyword and nothing else.

    Columns: kw_<theme> for each theme, kw_any (largest over all keywords), and has_note (0/1, an empty note gives zeros).
    """

    def __init__(self, keywords=None):
        self.keywords = keywords

    @staticmethod
    def clean(notes):
        notes = pd.Series(notes)
        out = []
        for text in notes.astype("string").fillna("").str.lower():
            for pattern in _BOILERPLATE:
                text = pattern.sub(" ", text)
            out.append(re.sub(r"\s+", " ", text).strip())
        return pd.Series(out, index=notes.index, dtype=object)

    def fit(self, notes, y=None):
        notes = self.clean(notes)
        keywords = KEYWORDS if self.keywords is None else self.keywords
        self.themes_ = list(keywords)
        self.keyword_list_ = [k for theme in self.themes_ for k in keywords[theme]]
        self.keyword_theme_ = np.array([i for i, theme in enumerate(self.themes_) for _ in keywords[theme]])
        self.vectorizer_ = TfidfVectorizer(analyzer=stems, sublinear_tf=True)
        self.vectorizer_.fit(list(notes[notes != ""]) + self.keyword_list_)       # training notes + keywords
        self.keyword_matrix_ = self.vectorizer_.transform(self.keyword_list_)
        self.feature_names_out_ = [f"kw_{t}" for t in self.themes_] + ["kw_any", "has_note"]
        return self

    def similarities(self, notes):
        """Cosine similarity of every note with every single keyword (a table notes x keywords)."""
        check_is_fitted(self, "vectorizer_")
        notes = self.clean(notes)
        sims = (self.vectorizer_.transform(notes) @ self.keyword_matrix_.T).toarray()    # all vectors have length 1
        return pd.DataFrame(sims, index=notes.index, columns=self.keyword_list_)

    def transform(self, notes):
        check_is_fitted(self, "vectorizer_")
        notes = self.clean(notes)
        sims = (self.vectorizer_.transform(notes) @ self.keyword_matrix_.T).toarray()
        out = {f"kw_{theme}": sims[:, self.keyword_theme_ == i].max(axis=1) for i, theme in enumerate(self.themes_)}
        out["kw_any"] = sims.max(axis=1)
        out["has_note"] = (notes != "").astype("int8").to_numpy()
        return pd.DataFrame(out, index=notes.index)

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "feature_names_out_")
        return np.array(self.feature_names_out_, dtype=object)


class SquirrelFeatures(BaseEstimator, TransformerMixin):
    """Cleans the raw sightings and builds the feature table (numbers, categories and 0/1 columns).

    Everything the transformer needs to remember (the list of distance columns and the highlight colors)
    is learned in `fit`, so it only ever sees the training data. `transform` never changes the table it receives.

    Settings (all can be tuned later):
      keep_day_level         keep the weather and calendar columns (one value per day, only 11 days)
      drop_hill_distance     drop geo_dist_hill_m (it repeats the latitude Y, correlation -0.99)
      log_height             use log(1 + feet) for the height above ground (strongly right-skewed)
      log_distances          use log(1 + metres) for the distances to named places (not strongly skewed, so off)
      use_nearest_place_kind keep the kind of the nearest named place within 150 m (water, green, ..., none)
      use_highlight_colors   add one 0/1 column per highlight fur color
      min_highlight_count    a highlight color needs at least this many training rows to get a column
      use_keyword_similarity add columns with the cosine similarity between the notes and the symptom keywords
                             (see KeywordSimilarity). Needs the TEXT_COLUMNS. Off by default
    """

    def __init__(self, keep_day_level=False, drop_hill_distance=True, log_height=True, log_distances=False,
                 use_nearest_place_kind=True, use_highlight_colors=True, min_highlight_count=10,
                 use_keyword_similarity=False):
        self.keep_day_level = keep_day_level
        self.drop_hill_distance = drop_hill_distance
        self.log_height = log_height
        self.log_distances = log_distances
        self.use_nearest_place_kind = use_nearest_place_kind
        self.use_highlight_colors = use_highlight_colors
        self.min_highlight_count = min_highlight_count
        self.use_keyword_similarity = use_keyword_similarity

    # ---------- helpers ----------
    @staticmethod
    def _frame(X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("SquirrelFeatures needs a pandas DataFrame with the raw sightings columns")
        return X

    def _needed_columns(self):
        needed = list(REQUIRED_COLUMNS) + list(getattr(self, "distance_columns_", []))
        if self.use_nearest_place_kind:
            needed.append("geo_nearest_place_kind")
        if self.keep_day_level:
            needed += DAY_LEVEL_NUMERIC + DAY_LEVEL_BINARY
        if self.use_keyword_similarity:
            needed += TEXT_COLUMNS
        return needed

    @staticmethod
    def _notes(X):
        """The free-text columns of a row joined into one note."""
        notes = X[TEXT_COLUMNS[0]].astype("string").fillna("")
        for col in TEXT_COLUMNS[1:]:
            notes = notes + " " + X[col].astype("string").fillna("")
        return notes

    def _build(self, X):
        out = {}

        # --- location: longitude / latitude as numbers ---
        out["X"] = X["X"].astype(float)
        out["Y"] = X["Y"].astype(float)

        # --- distance to the nearest named place of each kind (metres, always positive) ---
        for col in self.distance_columns_:
            v = X[col].astype(float)
            out[col] = np.log1p(v) if self.log_distances else v

        # --- height above ground: the column mixes numbers and the text 'FALSE' (= on the ground) ---
        raw = X["Above Ground Sighter Measurement"].astype("string")
        height = pd.to_numeric(raw, errors="coerce").astype(float)
        on_ground = raw.str.upper().eq("FALSE").fillna(False).astype(bool)      # missing stays missing
        height = height.mask(on_ground, 0.0)
        out["height_ft"] = np.log1p(height) if self.log_height else height   # still NaN where unknown
        out["height_missing"] = height.isna().astype("int8")

        # --- the 13 behavior tick boxes as 0/1 ---
        for b in BEHAVIORS:
            out[b] = X[b].fillna(False).astype(bool).astype("int8")

        # --- highlight fur color: one cell can hold several colors, e.g. "Gray, White" ---
        if self.use_highlight_colors:
            parts = X["Highlight Fur Color"].astype("string").fillna("").str.split(",")
            for color in self.highlight_colors_:
                out[f"highlight_{color.lower()}"] = parts.apply(
                    lambda ps, c=color: int(c in [p.strip() for p in ps])).astype("int8")

        # --- day-level columns (optional) ---
        if self.keep_day_level:
            for col in DAY_LEVEL_NUMERIC:
                out[col] = X[col].astype(float)
            out["wx_wind_missing"] = X["wx_wind_ms"].isna().astype("int8")
            out["cal_is_weekend"] = X["cal_is_weekend"].fillna(False).astype(bool).astype("int8")

        # --- keyword similarity of the notes (optional): floats between 0 and 1, left unscaled ---
        if self.use_keyword_similarity:
            out.update(self.keyword_.transform(self._notes(X)).to_dict("series"))

        # --- categories: unknown values get their own category instead of being dropped ---
        age = X["Age"].astype("string").str.strip()
        age = age.where(age.notna() & (age != "?"), "Unknown")                  # '?' and missing both mean unknown
        cats = {"Primary Fur Color": X["Primary Fur Color"].astype("string").fillna("Unknown"),
                "Age": age,
                "Location": X["Location"].astype("string").fillna("Unknown"),
                "Shift": X["Shift"].astype("string").fillna("Unknown")}
        if self.use_nearest_place_kind:
            cats["geo_nearest_place_kind"] = X["geo_nearest_place_kind"].astype("string").fillna("none")
        for name, s in cats.items():
            out[name] = s.astype(str).astype("category")

        return pd.DataFrame(out, index=X.index)

    # ---------- scikit-learn API ----------
    def fit(self, X, y=None):
        X = self._frame(X)
        missing = [c for c in REQUIRED_COLUMNS if c not in X.columns]
        if missing:
            raise KeyError(f"These columns are missing from the table: {missing}")

        # distance columns that exist in this data (a kind with no named places has no column)
        self.distance_columns_ = sorted(c for c in X.columns if c.startswith("geo_dist_")
                                        and not (c == "geo_dist_hill_m" and self.drop_hill_distance))
        # highlight colors seen in at least `min_highlight_count` TRAINING rows
        colors = (X["Highlight Fur Color"].dropna().astype(str).str.split(",").explode().str.strip())
        counts = colors.value_counts()
        self.highlight_colors_ = sorted(counts[counts >= self.min_highlight_count].index)

        absent = [c for c in self._needed_columns() if c not in X.columns]
        if absent:
            raise KeyError(f"These columns are missing from the table: {absent}")
        if self.use_keyword_similarity:
            self.keyword_ = KeywordSimilarity().fit(self._notes(X))           # IDF weights from the training notes only
        self.feature_names_out_ = list(self._build(X.head(5)).columns)
        return self

    def transform(self, X):
        check_is_fitted(self, "feature_names_out_")
        X = self._frame(X)
        absent = [c for c in self._needed_columns() if c not in X.columns]
        if absent:
            raise KeyError(f"These columns are missing from the table: {absent}")
        return self._build(X)

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "feature_names_out_")
        return np.array(self.feature_names_out_, dtype=object)


def make_preprocessor(keep_day_level=False, drop_hill_distance=True, log_height=True, log_distances=False,
                      use_nearest_place_kind=True, use_highlight_colors=True, min_highlight_count=10,
                      use_keyword_similarity=False, scaler="standard", impute_strategy="median", min_category_count=5):
    """Full preprocessing: clean and build features, fill missing numbers, scale numbers, one-hot the categories.

    scaler            'standard' (mean 0, std 1), 'robust' (median and IQR, less sensitive to outliers) or 'none'
    impute_strategy   how a missing number is filled: 'median' or 'mean'
    min_category_count  categories seen in fewer training rows are grouped together as 'infrequent'
    use_keyword_similarity  add the keyword similarity columns (kw_*, has_note). The kw_* columns are already between
                        0 and 1, so they are not scaled
    All fitting (means, medians, categories, highlight colors) happens on the table given to `fit`.
    """
    scalers = {"standard": StandardScaler(), "robust": RobustScaler(), "none": "passthrough"}
    if scaler not in scalers:
        raise ValueError(f"scaler must be one of {list(scalers)}")

    numbers = Pipeline([("fill", SimpleImputer(strategy=impute_strategy)),
                        ("scale", scalers[scaler])])
    categories = OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=min_category_count,
                               sparse_output=False)
    columns = ColumnTransformer(
        [("numbers", numbers, make_column_selector(dtype_include=np.floating, pattern=r"^(?!kw_)")),
         ("categories", categories, make_column_selector(dtype_include="category"))],
        remainder="passthrough",                      # the 0/1 columns stay as they are
        verbose_feature_names_out=False)

    pipe = Pipeline([("features", SquirrelFeatures(
        keep_day_level=keep_day_level, drop_hill_distance=drop_hill_distance,
        log_height=log_height, log_distances=log_distances,
        use_nearest_place_kind=use_nearest_place_kind, use_highlight_colors=use_highlight_colors,
        min_highlight_count=min_highlight_count, use_keyword_similarity=use_keyword_similarity)),
        ("columns", columns)])
    return pipe.set_output(transform="pandas")