import scrapy


class Article(scrapy.Item):
    url = scrapy.Field()
    date = scrapy.Field()
    category = scrapy.Field()
    title = scrapy.Field()
    author = scrapy.Field()
    intro = scrapy.Field()
    text = scrapy.Field()
    caption = scrapy.Field()
    keyword = scrapy.Field()
