"""
Hong Kong News Scraper.

Usage:
    scraper [-L LEVEL] [-S STARTDATE] OUTPUT
    scraper -h
    scraper -v

Options:
    -h              Show this screen.
    -v              Show version.
    -L LEVEL        Change how many text are printed. [default: WARNING]
    -S STARTDATE    Earliest date news are scraped. [default: 2018-01-01]
"""
from datetime import datetime as Datetime

from docopt import docopt
from scrapy.crawler import CrawlerProcess

from spiders.appledaily import AppleDailySpider
from spiders.rthk import RTHKSpider
from spiders.wenweipo import WenWeiPoSpider


def main():
    args = docopt(__doc__)
    start_date = Datetime.strptime(args["-S"], "%Y-%m-%d")

    process = CrawlerProcess(dict(
        FEED_URI=args["OUTPUT"],
        LOG_LEVEL=args["-L"],
        ROBOTSTXT_OBEY=True,
    ))

    process.crawl(AppleDailySpider, start_date=start_date)
    process.crawl(RTHKSpider, start_date=start_date)
    process.crawl(WenWeiPoSpider, start_date=start_date)
    process.start()


if __name__ == '__main__':
    main()
