import re
from datetime import date as Date
from datetime import datetime as Datetime

import scrapy
from dateutil.rrule import DAILY, rrule

from items import Article

date_re = re.compile(r"(\d{8})/\d+")


class AppleDailySpider(scrapy.Spider):
    name = "appledaily"

    def __init__(self, start_date=Date(2018, 1, 1), **kwargs):
        self.start_date = start_date
        super().__init__(**kwargs)

    def start_requests(self):
        dates = rrule(DAILY, dtstart=self.start_date, until=Date.today())
        for date in dates:
            url = f"https://hk.appledaily.com/catalog/index/{date:%Y%m%d}"
            yield scrapy.Request(url=url, callback=self.parse_link)

    def parse_link(self, response):
        items = response.css(
            "#tab1 div.title,"
            "#tab1 a:not(.icon_ArchiveVideo)"
        )

        category = None

        for item in items:
            url = item.css("::attr(href)").extract_first()
            if url is None:
                category = item.css("::text").extract_first()
            else:
                yield response.follow(
                    url,
                    callback=self.parse_article,
                    meta=dict(category=category),
                )

    def parse_article(self, response):
        url = response.url
        date = date_re.search(url).group(1)
        date = Datetime.strptime(date, "%Y%m%d")

        category = response.meta["category"]
        title = response.css(
            'meta[name="title"]::attr(content)').extract_first()

        keyword = response.css(
            'meta[name="keywords"][content]::attr(content)').extract_first()
        keyword = keyword if keyword is not None else ""

        content = response.css("#articleContent")

        intro = content.css(".ArticleIntro::text").extract()
        intro = "\n".join(intro)

        text = content.css(
            ".ArticleContent_Inner > :not(.ArticleIntro)::text").extract()
        text = "\n".join(text)

        caption = content.css("img[alt]::attr(alt)").extract()
        caption = "\n".join(caption)

        article = Article(
            url=url,
            date=date,
            category=category,
            title=title,
            author="",
            intro=intro,
            text=text,
            caption=caption,
            keyword=keyword,
        )

        yield article
