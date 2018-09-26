#!/usr/bin/env python3

import re
import sys
from collections import namedtuple
from datetime import datetime as Datetime
from itertools import groupby
from operator import itemgetter
from pathlib import Path
from typing import Set
from zipfile import ZipFile

import dash
import dash_core_components as dcc
import dash_html_components as html
import jieba
from dash.dependencies import Input, Output
from numpy import asarray, empty, expand_dims, unique
from pandas import DataFrame, read_csv
from plotly.offline import plot
from scipy.sparse import csr_matrix, find
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def normalize(X, axis):
    epsilon = sys.float_info.epsilon
    return X / (epsilon + expand_dims(X.sum(axis=axis), axis))


def doc_word_to_source_word(doc_word, data_source):
    encoder = CountVectorizer(token_pattern="^.+$", lowercase=False)

    doc_source = encoder.fit_transform(data_source)
    doc_source /= doc_source.sum(axis=0)

    source = encoder.get_feature_names()
    source_word = csr_matrix(doc_source.T) @ doc_word

    return source, source_word


def most_relevant_word(word, word_topic, n):
    t = word_topic.shape[1]
    word_topic = normalize(word_topic, axis=0)
    group = empty([t, n], dtype=int)
    index = empty([t, n], dtype=int)
    for r in range(t):
        group[r, :] = r
        index[r, :] = word_topic[:, r].argsort()[-n:]

    return group.ravel(), index.ravel()


def load_userdict(root: Path):
    root = Path(root)
    for userdict in root.iterdir():
        jieba.load_userdict(str(userdict.resolve()))


def load_stopword(root: Path):
    root = Path(root)
    for stopword in root.iterdir():
        with open(stopword, "r", encoding="utf8") as fin:
            for row in fin:
                yield row.strip()


Data = namedtuple("Data", [
    "source",
    "category",
    "subcategory",
    "date",
    "time",
    "article_id",
    "file",
    "path",
])
entry_re = re.compile(r"(\w+)_(\w+)_(\d{8}_\d{4})_(\d+)_\w+\.txt")


def load_info(file: Path):
    with ZipFile(file) as zin:
        for path in zin.namelist():
            match = entry_re.search(path)

            if match is None:
                continue

            category, subcategory, date, article_id = match.groups()

            source = Path(path).parent.stem.lower()

            category = category.lower()
            subcategory = subcategory.lower()

            datetime = Datetime.strptime(date, r"%Y%m%d_%H%M")
            date = datetime.date()
            time = datetime.time()
            article_id = int(article_id)

            yield Data(
                source,
                category,
                subcategory,
                date,
                time,
                article_id,
                file,
                path,
            )


def load_text(files, paths):
    for file, group in groupby(zip(files, paths), itemgetter(0)):
        with ZipFile(file) as zin:
            for _, path in group:
                yield zin.read(path).decode()


def ensure_non_zero(value, point_name, point_group):
    non_zero = unique(find(value)[1])
    value = value[:, non_zero].toarray()
    point_name = asarray([point_name[i] for i in non_zero])
    point_group = asarray([point_group[i] for i in non_zero])

    return value, point_name, point_group


def make_axis(title):
    return {
        "title": title,
        "tickangle": 0,
        "titlefont": {
            "size": 20
        },
        "tickfont": {
            "size": 15
        },
        "tickcolor": "rgba(0,0,0,0)",
        "ticklen": 5,
        "showline": True,
        "showgrid": True,
    }


def make_points(points, names, values, group, hue, type="markers"):
    a, b, c = points
    alpha = values / values.max()
    color = [f"hsla({hue}, 1, 0.5, {a})" for a in alpha]
    return {
        "type": "scatterternary",
        "text": names,
        "mode": type,
        "name": group,
        "marker": {
            "color": color,
        },
        "textfont": {
            "color": color,
        },
        "a": a,
        "b": b,
        "c": c,
    }


def plot_ternary(value, point_name, axis_name, point_group=None, type="markers", title=""):
    value, point_name, point_group = ensure_non_zero(
        value, point_name, point_group)
    axis_name = asarray(axis_name)

    subtotal = value.sum(axis=0)
    ratio = value * 100 / subtotal[None, :]
    intensity = subtotal / subtotal.max()

    if point_group is None:
        data = [make_points(ratio, point_name, intensity, None, 0)]
    else:
        gs = asarray(point_group)
        ugs = unique(gs)
        data = [
            make_points(
                ratio[:, g == gs],
                point_name[g == gs],
                intensity[g == gs],
                g,
                360 * i // len(ugs),
                type,
            )
            for i, g in enumerate(ugs)
        ]

    layout = {
        "ternary": {
            "sum": 100,
            **{
                f"{chr(97+i)}axis":
                make_axis(chr(65+i) + ": " + axis_name[i])
                for i in range(value.shape[0])
            }
        },
        "annotations": [{
            "showarrow": False,
            "text": title,
            "x": 0.5,
            "y": 1.3,
            "font": {"size": 15}
        }]
    }
    fig = {"data": data, "layout": layout}

    return fig


Context = namedtuple("Context", [
    "source_word",
    "word",
    "source",
    "word_topic",
])


def prepare_data(
    root: Path,
    sources: Set[str],
    categories: Set[str],
    subcategories: Set[str],
):
    root = Path(root)

    try:
        data = read_csv("data.csv.xz")
    except Exception:
        data = load_info(root/"raw.zip")
        data = (
            row for row in data
            if row.source in sources
            and row.category in categories
            and row.subcategory in subcategories
        )
        data = DataFrame(data)

        data.to_csv("data.csv.xz", index=False, compression="xz")

    stop_words = load_stopword(root/"stopwords_dict")
    stop_words = set(stop_words)

    try:
        vectorizer = joblib.load("vectorizer.pkl")
    except Exception:
        vectorizer = TfidfVectorizer(
            max_df=0.9,
            min_df=20,
            stop_words=stop_words,
        )

        vectorizer.fit(load_text(data.file, data.path))
        joblib.dump(vectorizer, "vectorizer.pkl")

    try:
        word = joblib.load("word.pkl")
        doc_word = joblib.load("doc_word.pkl")
    except Exception:
        word = vectorizer.get_feature_names()
        doc_word = vectorizer.transform(load_text(data.file, data.path))
        joblib.dump(word, "word.pkl")
        joblib.dump(doc_word, "doc_word.pkl")

    try:
        source = joblib.load("source.pkl")
        source_word = joblib.load("source_word.pkl")
    except Exception:
        source, source_word = doc_word_to_source_word(doc_word, data.source)
        joblib.dump(source, "source.pkl")
        joblib.dump(source_word, "source_word.pkl")

    try:
        featurizer = joblib.load("featurizer.pkl")
    except Exception:
        featurizer = LatentDirichletAllocation(
            n_components=7,
            learning_offset=20.,
            doc_topic_prior=0.53,
            topic_word_prior=0.2,
            max_iter=25,
            learning_method="online",
            batch_size=4096,
            verbose=1,
        )

        featurizer.fit(doc_word)
        joblib.dump(featurizer, "featurizer.pkl")

    try:
        doc_topic = joblib.load("doc_topic.pkl")
        word_topic = joblib.load("word_topic.pkl")
    except Exception:
        doc_topic = featurizer.transform(doc_word)
        word_topic = featurizer.components_.T
        joblib.dump(doc_topic, "doc_topic.pkl")
        joblib.dump(word_topic, "word_topic.pkl")

    return Context(source_word, word, source, word_topic)


def plot_source_word_ternery(
    sources: Set[str],
    n_result: int,
    type: str,
    context: Context,
):
    assert len(sources) == 3

    top_group, top_index = most_relevant_word(
        context.word, context.word_topic, n_result
    )

    source_index = asarray([list(context.source).index(s) for s in sources])

    source_name = {
                    "apple": "蘋果日報",
                    "wwp": "文滙報",
                    "oriental": "東方日報",
                    "mingpao": "明報",
    }

    topic_name = [
        "社會問題",
        "經濟",
        "企業",
        "科技",
        "法院",
        "旅遊",
        "外交",
    ]

    fig = plot_ternary(
        context.source_word[source_index, :][:, top_index],
        asarray([context.word[i] for i in top_index]),
        asarray([source_name[context.source[i]] for i in source_index]),
        # asarray([f"Topic {i + 1}" for i in top_group]),
        asarray([topic_name[i] for i in top_group]),
        type=type,
        title="",
    )

    return fig


def dash_ternery_app(context: Context):
    app = dash.Dash()
    app.title = 'NewsWatch Ternery Plot'

    app.layout = html.Div([
        html.Div([
            dcc.Dropdown(
                id="sources",
                options=[
                    {"label": "蘋果日報", "value": "apple"},
                    {"label": "文滙報", "value": "wwp"},
                    {"label": "東方日報", "value": "oriental"},
                    {"label": "明報", "value": "mingpao"},
                ],
                multi=True,
                value=["apple", "wwp", "oriental"],
            ),
            dcc.Dropdown(
                id="type",
                options=[
                    {"label": "Markers", "value": "markers"},
                    {"label": "Text", "value": "text"},
                ],
                multi=False,
                clearable=False,
                value=["markers"],
            ),
            dcc.Slider(
                id="n_result",
                min=0,
                max=200,
                step=1,
                value=100,
            ),
        ], className="row center-align", style={'maxWidth': '650px'}),
        html.Div([
            dcc.Graph(
                id="output_graph",
                style={'height': 600},
            )
        ], className="row center-align", style={'maxWidth': '650px'}),
    ], className='container')

    @app.callback(
        Output("output_graph", "figure"),
        [
            Input("sources", "value"),
            Input("type", "value"),
            Input("n_result", "value"),
        ]
    )
    def update_graph(sources, type, n_result):

        if isinstance(sources, str):
            sources = {sources}
        else:
            sources = set(sources)

        if not isinstance(type, str):
            type = "+".join(type)

        return plot_source_word_ternery(sources, n_result, type, context)

    return app


def main():
    context = prepare_data(
        Path(__file__).parent,
        {
            "apple", "mingpao", "oriental", "wwp",
        },
        {"news"},
        {"chinatw"},
    )
    app = dash_ternery_app(context)
    app.run_server(debug=False)


if __name__ == '__main__':
    main()
