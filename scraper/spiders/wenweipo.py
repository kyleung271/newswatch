from datetime import date as Date

import scrapy
from dateutil.rrule import DAILY, rrule

from items import Article


class WenWeiPoSpider(scrapy.Spider):
    name = "wenweipo"

    def start_requests(self, start_date=Date(2018, 1, 1)):
        dates = rrule(DAILY, dtstart=start_date, until=Date.today())
        for date in dates:
            url = f"http://pdf.wenweipo.com/{date:%Y/%m/%d}/pdf1.htm"
            yield scrapy.Request(url=url, callback=self.parse_page)

    def parse_page(self, response):
        urls = response.css(
            "div#colee1 a[href]:not(.a_1)::attr(href)").extract()
        for url in urls:
            yield response.follow(url, callback=self.parse_link)

    def parse_link(self, response):
        urls = response.css("div.pdf_c2br ::attr(href)").extract()
        for url in urls:
            yield response.follow(url, callback=self.parse_article)

    def parse_article(self, response):
        url = response.url

        title = response.css("h1.title::text").extract_first()
        date = response.css("span.date::text").extract_first()

        text = response.css(
            "div#main-content p:not(.connect-pdf)::text").extract()
        text = "\n".join(text)

        caption = response.css(
            "div#main-content img[alt]::attr(alt)").extract()
        caption = "\n".join(caption)

        category = response.css("span.current::text")
        category = category.re_first(r"\b(\w+)\s*>\s*正文")

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
        yield article
