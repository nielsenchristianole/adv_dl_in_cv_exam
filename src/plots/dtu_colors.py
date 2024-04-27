class DTUColors():
    def __init__(self):
        """Initialize the color dictionaries as defined in the Latex poster template."""
        self.secondary_color_dict = {
            "dtuyellow": "#FFCC00",
            "dtuorange": "#FF9900",
            "dtulightred": "#FF0000",
            "dtubrown": "#990000",
            "dtupurple": "#CC3399",
            "dtuviolet": "#660099",
            "dtudarkblue": "#3366CC",
            "dtulightblue": "#33CCFF",
            "dtulightgreen": "#99CC33",
            "dtudarkgreen": "#66CC00"
        }
        self.primary_color_dict = {
            "dtured": "#990000",
            "dtugrey": "#999999"
        }
    def get_secondary_color_dict(self):
        return self.secondary_color_dict
    def get_secondary_color(self, color_name):
        return self.secondary_color_dict[color_name]
    def get_primary_color_dict(self):
        return self.primary_color_dict
    def get_primary_color(self, color_name):
        return self.primary_color_dict[color_name]

