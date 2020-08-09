from scrapper.items import Blogger
import datetime
import scrapy

class BlogSpider(scrapy.Spider):
    name = "pyimagesearch-blog-spider"
    start_urls = ["https://fashionvignette.blogspot.com/"]

    def parse(self, response):
        urls = response.xpath('//div[@class="widget-content list-label-widget-content"]/ul/li')
        for url in urls:
            link = url.xpath('.//a/@href').extract_first()
            yield scrapy.Request(link, self.parse_page)

    def parse_page(self, response):
        subUrls = response.xpath('//div[@class="date-outer"]/div[@class="date-posts"]/div[@class="post-outer"]/div[@class="post hentry uncustomized-post-template"]/div[@class="post-body entry-content"]/div')
        for surl in subUrls:
            slink = surl.xpath('.//a/img/@src').extract_first()
            yield Blogger(file_urls=[slink])
              