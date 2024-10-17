
class Article:
    def __init__(self, id, pubtime, medium_code, medium_name, rubric, regional, doctype, doctype_description, language, char_count, dateline, head, subhead, article_link, content_id, content):
        self.id = id
        self.pubtime = pubtime
        self.medium_code = medium_code
        self.medium_name = medium_name
        self.rubric = rubric
        self.regional = regional
        self.doctype = doctype
        self.doctype_description = doctype_description
        self.language = language
        self.char_count = char_count
        self.dateline = dateline
        self.head = head
        self.subhead = subhead
        self.article_link = article_link
        self.content_id = content_id
        self.content = content