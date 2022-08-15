# coding=utf-8
import scrapy
import datetime

class AuthorSpider(scrapy.Spider):
    name = "happy"
    delta = datetime.timedelta(days=1)
    startday = datetime.date.today() - delta
    terminalday = datetime.date(2020,1,1)
    parseday = startday
    base_urls = 'http://paper.people.com.cn/rmrb/html/'
    timeline = (startday - terminalday).days

    def start_requests(self):
        url = self.base_urls + self.parseday.isoformat()[:-3] + "/" + self.parseday.isoformat()[8:10]+"/nbs.D110000renmrb_01.htm"
        yield scrapy.Request(url, self.parse)

    def parse(self, response):
        if self.timeline >= 0:
            page_links = self.base_urls + self.parseday.isoformat()[:-3] + "/" + self.parseday.isoformat()[8:10] + "/"
            next_pages = response.css("div.swiper-container div.swiper-slide a::attr(href)").getall()
            #找板块号
            for i in range(len(next_pages)):
                next_page = page_links + next_pages[i]
                yield scrapy.Request(next_page, callback=self.parsepassage,meta={'page_links':page_links})
            self.parseday -= self.delta
            self.timeline -= 1
        page_links = self.base_urls + self.parseday.isoformat()[:-3] + "/" + self.parseday.isoformat()[8:10] + "/"
        yield scrapy.Request(page_links+"nbs.D110000renmrb_01.htm",callback=self.parse)

    def parsepassage(self,response):
        base = response.meta['page_links']
        next_passages = response.css("ul.news-list li a::attr(href)").getall()
        for i in range(len(next_passages)):
            next_passage = base + next_passages[i]
            yield scrapy.Request(next_passage,callback=self.parsefinal)


    def parsefinal(self,response):
        for article in response.css('div.article'):
            yield {
                'title': article.css('h1::text').get(),
                'author': article.css('p.sec::text').get(),
                'contents': article.css('div#ozoom p::text').getall(),
            }
