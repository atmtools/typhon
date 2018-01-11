__all__ = ['XsecRecord']


class XsecRecord:
    def __init__(self, species=None, coeffs=None, fmin=None, fmax=None,
                 refpressure=None, xsec=None):
        self.version = 1
        self.species = species
        self.coeffs = coeffs
        self.fmin = fmin
        self.fmax = fmax
        self.refpressure = refpressure
        self.xsec = xsec

    def write_xml(self, xmlwriter, attr=None):
        """Write a XsecData object to an ARTS XML file.
        """
        # self.checksize()
        if attr is None:
            attr = {}
        attr['version'] = self.version
        xmlwriter.open_tag("XsecRecord", attr)
        xmlwriter.write_xml(self.species)
        xmlwriter.write_xml(self.coeffs)
        xmlwriter.write_xml(self.fmin)
        xmlwriter.write_xml(self.fmax)
        xmlwriter.write_xml(self.refpressure)
        xmlwriter.write_xml(self.xsec)
        xmlwriter.close_tag()

    @classmethod
    def from_xml(cls, xmlelement):
        """Loads a XsecData object from an xml.ElementTree.Element.
        """

        obj = cls()
        if 'version' in xmlelement.attrib.keys():
            obj.version = int(xmlelement.attrib['version'])
        else:
            obj.version = 1

        if obj.version != 1:
            raise RuntimeError(f'Unknown XsecRecord version {obj.version}')

        obj.species = xmlelement[0].value()
        obj.coeffs = xmlelement[1].value()
        obj.fmin = xmlelement[2].value()
        obj.fmax = xmlelement[3].value()
        obj.refpressure = xmlelement[4].value()
        obj.xsec = xmlelement[5].value()

        return obj
