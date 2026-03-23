from frappe.www.pages.base.base import BaseWebPage

class HelmetDetectionPage(BaseWebPage):
    def __init__(self, path, **kwargs):
        super().__init__(path, **kwargs)
        self.page_title = "Helmet Detection System"
        self.show_sidebar = False
        self.show_breadcrumbs = False

    def get_context(self, context):
        context.update({
            "title": "Helmet Detection System",
            "description": "AI-powered safety helmet detection for construction sites"
        })
        return context
