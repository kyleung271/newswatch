import re
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
import numpy
from dash.dependencies import Input, Output, State
from numpy import array
from pandas import DataFrame, read_csv
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances


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


def load_csv_mapping(file: Path):
    data = read_csv(file)
    keys = data[data.columns[0]]
    values = data[data.columns[1]]
    return dict(zip(keys, values))


def load_text(files, paths):
    for file, group in groupby(zip(files, paths), itemgetter(0)):
        with ZipFile(file) as zin:
            for _, path in group:
                yield zin.read(path).decode()


Context = namedtuple("Context", [
    "data",
    "doc_topic",
    "stop_words",
    "vectorizer",
    "featurizer",
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

        id_to_title = load_csv_mapping(root/"id_to_title.csv.xz")
        id_to_url = load_csv_mapping(root/"id_to_url.csv.xz")

        data["title"] = data.article_id.map(id_to_title)
        data["url"] = data.article_id.map(id_to_url)

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
        doc_word = joblib.load("doc_word.pkl")
    except Exception:
        doc_word = vectorizer.transform(load_text(data.file, data.path))
        joblib.dump(doc_word, "doc_word.pkl")

    try:
        featurizer = joblib.load("featurizer.pkl")
    except Exception:
        featurizer = LatentDirichletAllocation(
            n_components=45,
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
    except Exception:
        doc_topic = featurizer.transform(doc_word)
        joblib.dump(doc_topic, "doc_topic.pkl")

    return Context(data, doc_topic, stop_words, vectorizer, featurizer)


def euclidean_similarity(*args, **kwargs):
    return -euclidean_distances(*args, **kwargs)


def text_to_topic(
    text: str,
    stop_words: Set[str],
    vectorizer: TfidfVectorizer,
    featurizer: LatentDirichletAllocation,
):
    text = text.lower()
    text = re.sub(r"\d+", "num", " ".join(jieba.cut(text)))

    text = " ".join(
        word
        for word in text.split()
        if word not in stop_words
    )

    doc_word = vectorizer.transform([text])
    doc_topic = featurizer.transform(doc_word)

    return doc_topic


def similar_article(
    doc_topic_target,
    doc_topic,
    n_result=1,
    similarity=cosine_similarity,
):
    similarity = similarity(doc_topic, doc_topic_target).flatten()
    index = similarity.argsort()[::-1][:n_result]

    return index, similarity[index]


def recommendation(
        input_text: str,
        output_categories: Set[str],
        start_date: Datetime,
        end_date: Datetime,
        n_result: int,
        context: Context,
):
    is_selected = array(
        [c in output_categories for c in context.data.category])

    if start_date is not None:
        is_selected = is_selected & (context.data.date >= start_date)

    if end_date is not None:
        is_selected = is_selected & (context.data.date <= end_date)

    doc_topic_target = text_to_topic(
        input_text, context.stop_words, context.vectorizer, context.featurizer)

    index, similarity = similar_article(
        doc_topic_target,
        context.doc_topic[is_selected, :],
        n_result,
    )

    print(similarity)

    index = numpy.where(is_selected)[0][index]
    selected_data = context.data.iloc[list(index)]

    print(index)
    print(selected_data)

    yield from zip(selected_data.url, selected_data.title)


def dash_recommendation_app(context: Context):
    app = dash.Dash()
    app.title = 'NewsWatch Recommendation'
    app.layout = html.Div([
        html.Div([
            dcc.Textarea(
                id="text",
                placeholder="Paste an article here...",
                style={
                    "width": "100%"
                },
                rows=10,
            ),
        ], className="row center-align", style={'maxWidth': '650px'}),
        html.Div([
            dcc.Dropdown(
                id="category",
                options=[
                    {"label": "新聞", "value": "news"},
                    {"label": "評論", "value": "anacomm"},
                ],
                multi=True,
                value=["news", "anacomm"],
            ),
        ], className="row center-align", style={'maxWidth': '650px'}),
        html.Div([
            dcc.DatePickerRange(
                id='date-range',
                clearable=False,
                display_format='Y-M-D',
                min_date_allowed=Datetime(2018, 1, 1),
                start_date=Datetime(2018, 1, 1),
                end_date=Datetime.today(),
                start_date_placeholder_text="From",
                end_date_placeholder_text="Until",
            ),
        ], className="row center-align", style={'maxWidth': '650px'}),
        html.Div([
            html.Button(
                id="submit-button",
                type="submit",
                children="Submit",
                className="waves-effect btn-flat",
            ),
        ], className="row center-align", style={'maxWidth': '650px'}),
        html.Div([
            html.Div(
                id="output_div",
                className="center-align",
                style={'maxWidth': '650px'},
            ),
        ], className="row center-align", style={'maxWidth': '650px'}),
    ], className='container')

    @app.callback(
        Output("output_div", "children"),
        [
            Input("submit-button", "n_clicks")
        ],
        [
            State("text", "value"),
            State("category", "value"),
            State('date-range', 'start_date'),
            State('date-range', 'end_date'),
        ],
    )
    def update_output(clicks, text, categories, start_date, end_date):
        if clicks is None:
            return

        if isinstance(categories, str):
            categories = {categories}
        else:
            categories = set(categories)

        result = recommendation(
            text,
            categories,
            start_date,
            end_date,
            5,
            context,
        )

        return html.Div([
            html.Div([
                html.Div([
                    html.P(html.A(title, href=url, rel="external"))
                ], className="card-content")
            ], className="card")

            for url, title in result
        ])

    return app


def main():
    context = prepare_data(
        Path(__file__).parent,
        {
            "apple", "bas", "hk01", "mingpao",
            "oriental", "passion", "rthk", "wwp"
        },
        {"news", "anacomm"},
        {"local"},
    )

    app = dash_recommendation_app(context)
    app.run_server(debug=False)


if __name__ == "__main__":
    main()
