# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy


class MytimeItem(scrapy.Item):
    # define the fields for your item here like:
    # name = scrapy.Field()
    #quote = scrapy.Field()
    passage = scrapy.Field()
    title = scrapy.Field()
    contents = scrapy.Field()
    tag = scrapy.Field()
    pass
