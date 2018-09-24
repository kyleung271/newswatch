import re
from datetime import date as Date
from itertools import product

import scrapy
from dateutil.rrule import DAILY, rrule

from items import Article

category_re = re.compile(r'class="pathway">(\w+)</a>')


class RTHKSpider(scrapy.Spider):
    name = "rthk"

    def start_requests(self, start_date=Date(2018, 1, 1)):
        url = "http://news.rthk.hk/rthk/ch/news-archive.htm"
        dates = rrule(DAILY, dtstart=start_date, until=Date.today())

        for date, category in product(dates, range(7)):
            kwargs = dict(
                archive_year=str(date.year),
                archive_month=str(date.month),
                archive_day=str(date.day),
                archive_cat=str(category),
            )

            yield scrapy.FormRequest(
                url=url,
                method="GET",
                formdata=kwargs,
                callback=self.parse_link,
            )

    def parse_link(self, response):
        urls = response.css("span.title a[href]::attr(href)").extract()
        for url in urls:
            yield response.follow(
                url,
                callback=self.parse_article,
            )

    def parse_article(self, response):
        url = response.url
        category = category_re.search(response.text).group(1)

        date = response.css("div.createddate::text").extract_first()
        date = date.replace(" HKT ", " ")

        title = response.css("div.itemHeader .itemTitle::text").extract_first()

        text = response.css("div.itemFullText::text").extract()
        text = "\n".join(text)

        caption = response.css("div.itemBody img[alt]::attr(alt)").extract()
        caption = "\n".join(caption)

        article = Article(
            url=url,
            date=date,
            category=category,
            title=title,
            author="",
            intro="",
            text=text,
            caption=caption,
        )

        return article
